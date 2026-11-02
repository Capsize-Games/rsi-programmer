import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
reserve = 5 * 1024 * 1024
def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])
    return 0
if available() < 9 * 1024 * 1024:
    print('RSI regression reserve: below 9 GiB; not started.', flush=True)
    raise SystemExit(75)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
process = subprocess.Popen(sys.argv[1:], start_new_session=True)
try:
    while process.poll() is None:
        if available() < reserve:
            print('RSI regression reserve: below 5 GiB; stopping own job.', flush=True)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise SystemExit(75)
        time.sleep(0.25)
except BaseException:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
    raise
raise SystemExit(process.returncode)
