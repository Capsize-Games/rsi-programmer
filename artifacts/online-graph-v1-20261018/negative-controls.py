import copy
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

from scripts.audit_online_graph import audit_bundle

original = Path('artifacts/online-graph-v1-20261018')
manifest = json.loads((original / 'manifest.json').read_text())
lines = gzip.decompress((original / 'episodes.jsonl.gz').read_bytes())
lines = lines.decode('utf-8').splitlines()
first = json.loads(lines[0])
cases = ('features', 'reward', 'graph', 'missing_row', 'counter',
         'source_omission')
results = []
with tempfile.TemporaryDirectory(prefix='rsi-online-negative-') as directory:
    output = Path(directory)
    (output / 'summary.json').write_bytes(
        (original / 'summary.json').read_bytes())
    for name in cases:
        changed = copy.deepcopy(first)
        receipt = copy.deepcopy(manifest)
        if name == 'features':
            changed['attempts'][0]['features'][4] = 0.0
        elif name == 'reward':
            changed['attempts'][0]['reward'] += 0.25
        elif name == 'graph':
            changed['attempts'][0]['graph']['nodes'][0]['value'] = 99
        elif name == 'counter':
            changed['node_row_operations'] += 1
        elif name == 'source_omission':
            receipt['source_files'].pop('rsi_programmer/online_spiking.py')
        payload = lines[1:] if name == 'missing_row' else (
            [json.dumps(changed, sort_keys=True)] + lines[1:])
        raw = gzip.compress(('\n'.join(payload) + '\n').encode(), mtime=0)
        (output / 'episodes.jsonl.gz').write_bytes(raw)
        receipt['raw_sha256'] = hashlib.sha256(raw).hexdigest()
        (output / 'manifest.json').write_text(json.dumps(receipt))
        try:
            audit_bundle(output)
        except ValueError as error:
            results.append({'case': name, 'rejected': True,
                            'reason': str(error),
                            'raw_hash_recomputed': True})
        else:
            raise SystemExit('Corrupt bundle unexpectedly accepted: ' + name)
report = {'status': 'passed', 'controls': results,
          'scope': 'Six corrupted copies of the complete measured bundle; '
                   'checksums recomputed to test semantic/provenance checks'}
(original / 'negative-controls.json').write_text(
    json.dumps(report, indent=2, sort_keys=True) + '\n')
print(json.dumps(report, sort_keys=True))
