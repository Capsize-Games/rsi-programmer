import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
binary = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
before = digest(binary)
started = time.monotonic()
records = []
for path in sorted((root / 'curlee').glob('*.curlee')):
    record = {'program': path.name}
    for stage in ('check', 'run'):
        command = [str(binary), stage]
        if stage == 'run':
            command += ['--fuel', '100000']
        command.append(str(path))
        begin = time.monotonic()
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=5)
            record[stage] = {'exit_code': result.returncode,
                'stdout': result.stdout.replace(str(root), '<programs>'),
                'stderr': result.stderr.replace(str(root), '<programs>'),
                'wall_seconds': time.monotonic() - begin}
        except subprocess.TimeoutExpired:
            record[stage] = {'timeout': True, 'wall_seconds': 5}
            break
        if result.returncode:
            break
    record['passed'] = (record.get('check', {}).get('exit_code') == 0
        and record.get('run', {}).get('exit_code') == 0
        and bool(re.search(r'curlee run: result 0\b',
                           record.get('run', {}).get('stdout', ''))))
    records.append(record)
    print(json.dumps({'program': path.name, 'passed': record['passed']}),
          flush=True)
after = digest(binary)
report = {'scope': 'Actual CURLEE check and VM run of learned arithmetic functions, each with 36 observed and 24 held assertions in main. This does not qualify a general language programmer or compiler.',
          'compiler_sha256_before': before,
          'compiler_sha256_after': after,
          'compiler_stable': before == after,
          'programs': len(records),
          'passed_programs': sum(record['passed'] for record in records),
          'observed_cases_per_program': 36,
          'held_cases_per_program': 24,
          'wall_seconds': time.monotonic() - started,
          'records': records}
(root / 'curlee-execution.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: report[k] for k in ('compiler_stable', 'programs',
                                      'passed_programs', 'wall_seconds')}),
      flush=True)
raise SystemExit(0 if before == after and all(r['passed'] for r in records)
                 and len(records) == 78 else 1)
