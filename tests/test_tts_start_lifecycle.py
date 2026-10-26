"""TTS start reservations serialize filesystem checks and recover after spawn failure."""
from pathlib import Path
import threading
import subprocess

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
    monkeypatch.setattr(books, '_tts_processes', {}, raising=False)
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


class Process:
    def __init__(self):
        self.alive = True
        self.waited = False
        self.unkillable = False

    def poll(self):
        return None if self.alive else 1

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        if self.unkillable:
            raise subprocess.TimeoutExpired('fixture-tts', timeout)
        self.waited = True
        self.alive = False
        return 0


@pytest.fixture
def processes(monkeypatch):
    launched = []

    def launch(*args, **kwargs):
        process = Process()
        launched.append(process)
        return process

    monkeypatch.setattr(books.subprocess, 'Popen', launch)
    monkeypatch.setattr(books.subprocess, 'run', lambda *a, **k: None)
    return launched


def test_failed_progress_does_not_override_a_live_child(client, auth_headers, lock, processes):
    assert client.post('/api/tts/start/Example', headers=auth_headers).status_code == 200
    lock.write_text('-1')
    assert client.post('/api/tts/start/Example', headers=auth_headers).status_code == 409
    assert len(processes) == 1


def test_stop_reaps_owned_child_before_restart(client, auth_headers, lock, processes):
    assert client.post('/api/tts/start/Example', headers=auth_headers).status_code == 200
    assert client.post('/api/tts/stop/Example', headers=auth_headers).status_code == 200
    assert processes[0].waited and not processes[0].alive
    assert client.post('/api/tts/start/Example', headers=auth_headers).status_code == 200
    assert len(processes) == 2


def test_dead_owned_child_does_not_leave_a_stuck_reservation(client, auth_headers, lock, processes):
    assert client.post('/api/tts/start/Example', headers=auth_headers).status_code == 200
    processes[0].alive = False  # simulate a hard crash before progress cleanup
    assert client.post('/api/tts/start/Example', headers=auth_headers).status_code == 200
    assert len(processes) == 2


def test_unconfirmed_stop_cannot_allow_a_second_child(client, auth_headers, lock, processes):
    assert client.post('/api/tts/start/Example', headers=auth_headers).status_code == 200
    processes[0].unkillable = True
    assert client.post('/api/tts/stop/Example', headers=auth_headers).status_code == 500
    assert client.post('/api/tts/start/Example', headers=auth_headers).status_code == 409
    assert len(processes) == 1


def test_unowned_job_is_not_signalled_or_unreserved(client, auth_headers, lock, monkeypatch):
    lock.write_text('20')
    signals = []
    monkeypatch.setattr(books.subprocess, 'run', lambda *a, **k: signals.append(a))
    response = client.post('/api/tts/stop/Example', headers=auth_headers)
    assert response.status_code == 409
    assert lock.read_text() == '20'
    assert signals == []


def test_stop_without_a_job_is_a_noop(client, auth_headers, lock, monkeypatch):
    signals = []
    monkeypatch.setattr(books.subprocess, 'run', lambda *a, **k: signals.append(a))
    assert client.post('/api/tts/stop/Example', headers=auth_headers).status_code == 200
    assert not lock.exists()
    assert signals == []


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
