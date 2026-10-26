"""A refused stop must keep the audio job visible and polling."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(not shutil.which('node'), reason='Node is only needed for frontend checks')
@pytest.mark.parametrize('ok', [True, False])
def test_stop_controls_follow_the_server_result(ok):
    source = Path(__file__).resolve().parents[1] / 'static/js/modules/audiobooks.js'
    script = r'''
const fs = require('node:fs'), vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
const start = source.indexOf('async function stopGeneration() {');
const stop = source.indexOf('async function previewVoice()', start);
const code = source.slice(start, stop);
const ok = process.argv[2] === 'true', calls = [];
const context = {
  settingsBookId: 'Example', currentBook: 'Example', pollInterval: 7, isGenerating: true,
  fetch: async () => ({ok, json: async () => ok ? {status:'stopped'} : {error:'still running'}}),
  showToast: message => calls.push(message), friendlyError: e => e.message,
  clearInterval: () => calls.push('clear'), hideGenBanner: () => calls.push('hide'),
  loadBooks: () => calls.push('load'), closeSettings: () => calls.push('close')
};
(async () => {
  await vm.runInNewContext(code + '\nstopGeneration()', context);
  process.stdout.write(JSON.stringify({calls, pollInterval:context.pollInterval}));
})().catch(e => { console.error(e); process.exitCode = 1; });
'''
    result = subprocess.run([shutil.which('node'), '-e', script, str(source), str(ok).lower()],
                            capture_output=True, text=True, timeout=10, check=True)
    data = json.loads(result.stdout)
    if ok:
        assert data == {'calls': ['clear', 'hide', 'load', 'close'], 'pollInterval': None}
    else:
        assert data == {'calls': ['still running'], 'pollInterval': 7}
