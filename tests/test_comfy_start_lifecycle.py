"""An owned ComfyUI process is already starting before HTTP is ready."""
import pytest
import threading

from app import app
import routes.comfy as comfy


class Process:
    pid = 12345

    def __init__(self):
        self.alive = True
        self.terminated = False

    def poll(self):
        return None if self.alive else 1

    def terminate(self):
        self.terminated = True
        self.alive = False

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def processes(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, '_comfy_process', None, raising=False)
    monkeypatch.setattr(comfy, '_comfy_online', lambda: False)
    monkeypatch.setattr(comfy, '_comfy_strategy', lambda: ('cmd', ['fixture-comfy']))
    monkeypatch.setattr(comfy, 'COMFY_PID_FILE', tmp_path / 'comfy.pid')
    monkeypatch.setattr(comfy, 'COMFY_LOG_FILE', tmp_path / 'comfy.log')
    monkeypatch.setattr(comfy.os, 'kill', lambda *args: None)
    launched = []

    def launch(*args, **kwargs):
        proc = Process()
        launched.append(proc)
        return proc

    monkeypatch.setattr(comfy.subprocess, 'Popen', launch)
    return launched


def test_retry_during_warmup_launches_once(client, auth_headers, processes):
    first = client.post('/api/comfy/start', headers=auth_headers)
    retry = client.post('/api/comfy/start', headers=auth_headers)
    assert first.status_code == retry.status_code == 200
    assert len(processes) == 1


def test_exited_process_can_be_restarted(client, auth_headers, processes):
    assert client.post('/api/comfy/start', headers=auth_headers).status_code == 200
    processes[0].alive = False
    assert client.post('/api/comfy/start', headers=auth_headers).status_code == 200
    assert len(processes) == 2


def test_stop_reaps_owned_child(client, auth_headers, processes):
    assert client.post('/api/comfy/start', headers=auth_headers).status_code == 200
    assert client.post('/api/comfy/stop', headers=auth_headers).status_code == 200
    assert processes[0].terminated
    assert comfy._comfy_process is None
    assert not comfy.COMFY_PID_FILE.exists()


def test_concurrent_requests_reserve_launch_before_popen_returns(
        auth_headers, processes, monkeypatch):
    entered, release, second_launch = threading.Event(), threading.Event(), threading.Event()
    launches, results, errors = [], [], []

    def delayed_launch(*args, **kwargs):
        proc = Process()
        launches.append(proc)
        if len(launches) == 1:
            entered.set()
            assert release.wait(5)
        else:
            second_launch.set()
        return proc

    monkeypatch.setattr(comfy.subprocess, 'Popen', delayed_launch)

    def start():
        try:
            with app.test_client() as client:
                results.append(client.post('/api/comfy/start', headers=auth_headers).status_code)
        except BaseException as exc:
            errors.append(exc)

    first, second = threading.Thread(target=start), threading.Thread(target=start)
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        second_launch.wait(0.2)
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not errors, errors
    assert not first.is_alive() and not second.is_alive()
    assert results == [200, 200]
    assert len(launches) == 1
