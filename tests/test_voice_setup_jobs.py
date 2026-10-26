"""Queued setup jobs reserve their operation before the worker runs."""
import pytest
from types import SimpleNamespace

import routes.voice as voice


CASES = [('/api/voice/ollama/install', {}),
         ('/api/voice/ollama/pull', {'model': 'tiny:latest'}),
         ('/api/voice/tts/install', {'engine': 'piper-kathleen-low'})]


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(voice, '_setup_jobs', {})
    monkeypatch.setattr(voice.shutil, 'which', lambda name: '/fixture/ollama')


@pytest.mark.parametrize('url,body', CASES)
def test_retry_before_worker_runs_dispatches_once(client, auth_headers, monkeypatch, url, body):
    calls = []

    class DeferredThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            calls.append(self.kwargs)

    monkeypatch.setattr(voice, 'threading', SimpleNamespace(Thread=DeferredThread))
    first = client.post(url, json=body, headers=auth_headers)
    retry = client.post(url, json=body, headers=auth_headers)
    assert first.status_code == retry.status_code == 202
    assert first.json['job'] == retry.json['job']
    assert len(calls) == 1


@pytest.mark.parametrize('url,body', CASES)
def test_dispatch_failure_is_retryable(client, auth_headers, monkeypatch, url, body):
    calls = []

    class Thread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError('simulated thread exhaustion')

    monkeypatch.setattr(voice, 'threading', SimpleNamespace(Thread=Thread))
    from app import app
    monkeypatch.setitem(app.config, 'PROPAGATE_EXCEPTIONS', False)
    failed = client.post(url, json=body, headers=auth_headers)
    assert failed.status_code == 500
    assert next(iter(voice._setup_jobs.values()))['state'] == 'error'
    retry = client.post(url, json=body, headers=auth_headers)
    assert retry.status_code == 202
    assert len(calls) == 2
