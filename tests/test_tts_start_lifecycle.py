"""TTS start reservations serialize filesystem checks and recover after spawn failure."""
from pathlib import Path
import threading

import pytest

from app import app
import routes.books as books
import shared


@pytest.fixture
def lock(tmp_path, monkeypatch):
    source, audio = tmp_path / 'books', tmp_path / 'audio'
    source.mkdir()
    audio.mkdir()
    (source / 'Example.json').write_text('[]')
    monkeypatch.setattr(shared, 'BOOKS_DIR', source)
    monkeypatch.setattr(books, 'AUDIO_DIR', audio)
    return audio / 'Example.tts_lock'


def test_overlapping_restart_after_error_launches_once(auth_headers, lock, monkeypatch):
    lock.write_text('-1')
    first_read, second_read, release = threading.Event(), threading.Event(), threading.Event()
    original_read = Path.read_text
    readers, launches, responses, errors = [], [], [], []

    def slow_read(path, *args, **kwargs):
        value = original_read(path, *args, **kwargs)
        if path == lock:
            readers.append(threading.current_thread().name)
            if len(readers) == 1:
                first_read.set()
                assert release.wait(5)
            else:
                second_read.set()
        return value

    monkeypatch.setattr(Path, 'read_text', slow_read)
    monkeypatch.setattr(books.subprocess, 'Popen', lambda *a, **k: launches.append(a))

    def start():
        try:
            with app.test_client() as client:
                responses.append(client.post('/api/tts/start/Example', headers=auth_headers).status_code)
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=start)
    second = threading.Thread(target=start)
    first.start()
    try:
        assert first_read.wait(5)
        second.start()
        second_read.wait(0.2)  # with the fix the second read waits for the start lock
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors, errors
    assert sorted(responses) == [200, 409]
    assert len(launches) == 1


def test_failed_spawn_allows_retry(client, auth_headers, lock, monkeypatch):
    monkeypatch.setitem(app.config, 'PROPAGATE_EXCEPTIONS', False)

    def fail(*args, **kwargs):
        raise OSError('simulated process exhaustion')

    monkeypatch.setattr(books.subprocess, 'Popen', fail)
    failed = client.post('/api/tts/start/Example', headers=auth_headers)
    assert failed.status_code == 500
    launches = []
    monkeypatch.setattr(books.subprocess, 'Popen', lambda *a, **k: launches.append(a))
    retry = client.post('/api/tts/start/Example', headers=auth_headers)
    assert retry.status_code == 200
    assert len(launches) == 1
