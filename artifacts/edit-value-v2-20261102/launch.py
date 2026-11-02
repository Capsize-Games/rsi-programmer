import os
from pathlib import Path
import subprocess
import sys
import time
started=time.monotonic()
while time.monotonic()-started<600:
    entries=Path('/proc/meminfo').read_text().splitlines()
    available=int(next(s for s in entries if s.startswith('MemAvailable:')).split()[1])
    if available>=9*1024*1024:
        command=['timeout','1200','python3','/tmp/rsi-run-with-reserve.py','flock','/tmp/rsi-programmer-builder.lock','.venv/bin/python','-m','scripts.measure_edit_value','artifacts/edit-value-v2-20261102','--budget','5000','--search-seconds','3','--training-tasks','64','--transitions','2048','--epochs','30','--rollouts','2','--rollout-steps','32']
        print('Resource admission accepted; starting frozen trial.',flush=True)
        raise SystemExit(subprocess.run(command,check=False).returncode)
    print('Waiting for 9 GiB admission; available KiB='+str(available),flush=True)
    time.sleep(10)
print('Resource admission timed out; no learning result.',flush=True)
raise SystemExit(75)
