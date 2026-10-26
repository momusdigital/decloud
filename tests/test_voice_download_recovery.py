"""Interrupted HTTP transfers must not publish installable voice files."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading

import pytest

import routes.voice as voice


@pytest.fixture
def server():
    state = {'drop': '.onnx', 'requests': []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state['requests'].append(self.path)
            data = b'{"voice":"fixture"}' if self.path.endswith('.json') else b'M' * 32768
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            if state['drop'] and self.path.endswith(state['drop']):
                self.wfile.write(data[:max(1, len(data) // 2)])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
            else:
                self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_port}/voice', state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)


@pytest.fixture
def model(tmp_path, monkeypatch, server):
    base, _ = server
    target = tmp_path / 'voice.onnx'
    monkeypatch.setattr(voice, '_setup_jobs', {})
    monkeypatch.setattr(voice, 'TTS_ENGINES', {
        'fixture': {'engine': 'piper', 'model': str(target)}})
    monkeypatch.setattr(voice, 'VOICE_MODEL_URLS', {'fixture': base})
    return target


@pytest.mark.parametrize('drop', ['.onnx', '.json'])
def test_interrupted_transfer_is_not_published_and_retry_completes(model, server, drop):
    _, state = server
    state['drop'] = drop
    voice._run_voice_download('job', 'fixture')
    assert voice._setup_jobs['job']['state'] == 'error'
    assert not model.exists()
    assert not voice._tts_installed('fixture')
    assert list(model.parent.iterdir()) == []

    state['drop'] = None
    voice._run_voice_download('job', 'fixture')
    assert voice._setup_jobs['job']['state'] == 'done'
    assert model.read_bytes() == b'M' * 32768
    assert json.loads(Path(str(model) + '.json').read_text()) == {'voice': 'fixture'}
    assert voice._tts_installed('fixture')


def test_installed_voice_is_reused_without_downloading(model, server):
    model.write_bytes(b'working model')
    config = Path(str(model) + '.json')
    config.write_text('{"voice":"old"}')
    voice._run_voice_download('job', 'fixture')
    assert voice._setup_jobs['job']['state'] == 'done'
    assert server[1]['requests'] == []
    assert model.read_bytes() == b'working model'
    assert config.read_text() == '{"voice":"old"}'
    assert sorted(p.name for p in model.parent.iterdir()) == ['voice.onnx', 'voice.onnx.json']


def test_publication_failure_cannot_change_a_working_voice(model, server, monkeypatch):
    model.write_bytes(b'working model')
    config = Path(str(model) + '.json')
    config.write_text('{"voice":"old"}')
    server[1]['drop'] = None
    replace = os.replace

    def fail_model_publish(source, destination):
        if str(destination) == str(model):
            raise PermissionError('simulated model file in use')
        return replace(source, destination)

    monkeypatch.setattr(voice.os, 'replace', fail_model_publish)
    voice._run_voice_download('job', 'fixture')
    assert model.read_bytes() == b'working model'
    assert config.read_text() == '{"voice":"old"}'


def test_repair_missing_config_does_not_replace_existing_model(model, server):
    model.write_bytes(b'working model')
    voice._run_voice_download('job', 'fixture')
    assert voice._setup_jobs['job']['state'] == 'done'
    assert model.read_bytes() == b'working model'
    assert server[1]['requests'] == ['/voice.onnx.json']
    assert voice._tts_installed('fixture')


def test_repair_missing_model_does_not_replace_existing_config(model, server):
    config = Path(str(model) + '.json')
    config.write_text('{"voice":"old"}')
    server[1]['drop'] = None
    voice._run_voice_download('job', 'fixture')
    assert voice._setup_jobs['job']['state'] == 'done'
    assert config.read_text() == '{"voice":"old"}'
    assert server[1]['requests'] == ['/voice.onnx']
    assert voice._tts_installed('fixture')


def test_model_without_config_is_not_installed(model):
    model.write_bytes(b'complete model')
    assert not voice._tts_installed('fixture')


@pytest.mark.skipif(os.name == 'nt' or not shutil.which('bash'), reason='POSIX installer')
def test_installer_retries_failed_transfer_and_skips_completed_files(tmp_path):
    installer = (Path(__file__).resolve().parents[1] / 'install.sh').read_text()
    helper = installer[installer.index('_fetch_voice() {'):]
    helper = helper[:helper.index('\n}\n') + 3]
    binaries, piper = tmp_path / 'bin', tmp_path / 'piper'
    binaries.mkdir()
    piper.mkdir()
    curl = binaries / 'curl'
    curl.write_text('''#!/bin/sh
while [ "$#" -gt 0 ]; do
    if [ "$1" = '-o' ]; then shift; target="$1"; fi
    shift
done
printf '%s\\n' "$target" >> "$FIXTURE_CALLS"
printf '%s' "$FIXTURE_BODY" > "$target"
exit "$FIXTURE_STATUS"
''')
    curl.chmod(0o755)
    calls = tmp_path / 'calls.log'
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ['PATH'],
               PIPER_DIR=str(piper), FIXTURE_CALLS=str(calls),
               FIXTURE_BODY='partial', FIXTURE_STATUS='18')

    def run():
        subprocess.run(['bash', '-c', helper + '\n_fetch_voice lessac high\n'],
                       env=env, capture_output=True, text=True, check=True, timeout=10)

    run()
    assert list(piper.iterdir()) == []
    env.update(FIXTURE_STATUS='0', FIXTURE_BODY='complete')
    run()
    assert (piper / 'en_US-lessac-high.onnx').read_bytes() == b'complete'
    assert (piper / 'en_US-lessac-high.onnx.json').read_bytes() == b'complete'
    completed_calls = calls.read_text()
    run()
    assert calls.read_text() == completed_calls
    assert len(list(piper.iterdir())) == 2
