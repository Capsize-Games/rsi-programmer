import json
import runpy
namespace = runpy.run_path('/work/programs.py')
cases = json.load(open('/work/cases.json'))
failed = []
observed = held = passed = 0
for case in cases:
    ok = True
    for split in ('observed', 'held'):
        for x, y, expected in case[split]:
            actual = namespace[case['name']](x, y)
            if split == 'observed':
                observed += 1
            else:
                held += 1
            if actual != expected:
                ok = False
                failed.append({'function': case['name'], 'split': split,
                               'input': [x, y], 'expected': expected,
                               'actual': actual})
    passed += int(ok)
print(json.dumps({'programs': len(cases), 'passed_programs': passed,
                  'observed_cases': observed, 'held_cases': held,
                  'failed_cases': len(failed), 'failures': failed[:5]}))
raise SystemExit(0 if not failed else 1)
