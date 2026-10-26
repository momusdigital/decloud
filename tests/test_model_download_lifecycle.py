"""Retry, target ownership, and failure recovery for model downloads."""
import pytest
from types import SimpleNamespace

import routes.models as models


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(models, 'CHAT_DIR', tmp_path)
    monkeypatch.setattr(models, '_downloads', {})
    monkeypatch.setattr(models, 'get_specs', lambda: {})
    monkeypatch.setattr(models, 'analyze_repo', lambda *args: {'files': [{
        'file': 'model.gguf', 'parts': None, 'tier': 'runs-well', 'size_gb': 0.01,
    }]})


@pytest.fixture
def dispatch(monkeypatch):
    calls = []

    class DeferredThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            calls.append(self.kwargs)

    monkeypatch.setattr(models, 'threading', SimpleNamespace(Thread=DeferredThread))
    return calls


BODY = {'repo': 'owner/model', 'file': 'model.gguf', 'category': 'chat',
        'import_ollama': False}


def test_retry_reuses_queued_job(client, auth_headers, dispatch):
    first = client.post('/api/models/download', json=BODY, headers=auth_headers)
    retry = client.post('/api/models/download', json=BODY, headers=auth_headers)
    assert first.status_code == retry.status_code == 200
    assert retry.json['download_id'] == first.json['download_id']
    assert len(dispatch) == 1
    listing = client.get('/api/models/downloads', headers=auth_headers)
    assert listing.status_code == 200
    assert not any(k.startswith('_') for k in listing.json['downloads'][0])


def test_other_repo_cannot_write_active_target(client, auth_headers, dispatch):
    first = client.post('/api/models/download', json=BODY, headers=auth_headers)
    conflict = client.post('/api/models/download', json={**BODY, 'repo': 'other/model'},
                           headers=auth_headers)
    assert first.status_code == 200
    assert conflict.status_code == 409
    assert len(dispatch) == 1


@pytest.mark.parametrize('state', ['done', 'error', 'cancelled'])
def test_terminal_job_can_be_retried(client, auth_headers, dispatch, state):
    first = client.post('/api/models/download', json=BODY, headers=auth_headers)
    models._downloads[first.json['download_id']]['state'] = state
    retry = client.post('/api/models/download', json=BODY, headers=auth_headers)
    assert retry.status_code == 200
    assert retry.json['download_id'] != first.json['download_id']
    assert len(dispatch) == 2


def test_failed_thread_start_releases_target(client, auth_headers, monkeypatch):
    class BrokenThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError('cannot start thread')

    monkeypatch.setattr(models, 'threading', SimpleNamespace(Thread=BrokenThread))
    from app import app
    monkeypatch.setitem(app.config, 'PROPAGATE_EXCEPTIONS', False)
    first = client.post('/api/models/download', json=BODY, headers=auth_headers)
    assert first.status_code == 500
    assert list(models._downloads.values())[0]['state'] == 'error'
    monkeypatch.setattr(BrokenThread, 'start', lambda self: None)
    retry = client.post('/api/models/download', json=BODY, headers=auth_headers)
    assert retry.status_code == 200
    assert retry.json['download_id'] != first.json['download_id']


def test_cancelling_job_keeps_its_target_reserved(client, auth_headers, dispatch):
    first = client.post('/api/models/download', json=BODY, headers=auth_headers)
    models._downloads[first.json['download_id']]['state'] = 'cancelling'
    conflict = client.post('/api/models/download', json={**BODY, 'repo': 'other/model'},
                           headers=auth_headers)
    assert conflict.status_code == 409
    assert len(dispatch) == 1


class Stream:
    headers = {}

    def __init__(self, data=b'complete', fail=False):
        self.data, self.fail = data, fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, **kwargs):
        yield self.data
        if self.fail:
            raise OSError('interrupted stream')


def test_failed_download_preserves_existing_model(tmp_path, monkeypatch):
    final = tmp_path / 'model.gguf'
    final.write_bytes(b'working model')
    models._downloads['job'] = {'state': 'queued'}
    monkeypatch.setattr(models._requests, 'get', lambda *a, **k: Stream(fail=True))
    models._download_worker('job', 'owner/model', ['model.gguf'], 'chat', False)
    assert models._downloads['job']['state'] == 'error'
    assert final.read_bytes() == b'working model'
    assert list(tmp_path.iterdir()) == [final]


def test_cancelled_queued_job_does_not_download(monkeypatch):
    models._downloads['job'] = {'state': 'cancelling'}

    def unexpected(*args, **kwargs):
        pytest.fail('a cancelled queued job attempted a download')

    monkeypatch.setattr(models._requests, 'get', unexpected)
    models._download_worker('job', 'owner/model', ['model.gguf'], 'chat', False)
    assert models._downloads['job']['state'] == 'cancelled'


def test_split_download_publishes_only_complete_merge(tmp_path, monkeypatch):
    parts = ['model-00001-of-00002.gguf', 'model-00002-of-00002.gguf']
    final = tmp_path / 'model.gguf'
    final.write_bytes(b'old')
    models._downloads['job'] = {'state': 'queued'}

    def get(url, **kwargs):
        assert final.read_bytes() == b'old'
        return Stream(b'one' if parts[0] in url else b'two')

    monkeypatch.setattr(models._requests, 'get', get)
    models._download_worker('job', 'owner/model', parts, 'chat', False)
    assert models._downloads['job']['state'] == 'done'
    assert final.read_bytes() == b'onetwo'
    assert list(tmp_path.iterdir()) == [final]
