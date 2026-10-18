import json, math, tempfile, sys, time, copy, os
from pathlib import Path
from unittest.mock import patch
ROOT=Path(os.environ.get('REVIEW_REPO',Path.cwd())).resolve()
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
import torch
from rsi_programmer import author, ltrain, lprecision, modules, lattice, policy, promote, probe
from rsi_programmer.sandbox import SandboxRunner
from rsi_programmer.lineage import LineageStore

torch.set_num_threads(1)
def emit(name,**d):print(name,json.dumps(d,sort_keys=True,default=str),flush=True)
def packs():
 gs=author.contrastive_groups('fixtures/rsi_shadow_tasks.json');counts=[]
 for g in gs:
  m,w,b=ltrain.pack_rows(g)
  counts.append({'rows':len(g),'unique_bit_targets':[len(w[:,i].unique()) for i in range(4)],'membership_classes':len(b.unique())})
 emit('real_lattice_packs',packs=len(gs),mixed_bit_terms=sum(n>1 for d in counts for n in d['unique_bit_targets']),total_bit_terms=4*len(gs),detail=counts)
 # All grids tie: current threshold is discarded (doc only promises defaults).
 lat=lattice.BitLattice();lat.thresholds={**lat.thresholds,'b0':.55,'st':.1}
 with patch.object(lprecision,'_word_exact',return_value=1.),patch.object(lprecision,'_struct_f1',return_value=1.):
  rep=lprecision.sweep_thresholds(lat,(torch.zeros(2,13),torch.zeros(2,4),torch.zeros(2)))
 emit('sweep_tie',b0_after=lat.thresholds['b0'],st_after=lat.thresholds['st'])
def formula():
 gaps=torch.zeros(2,2,requires_grad=True);m=torch.nn.functional.softplus(1-gaps).mean();m.backward()
 emit('margin_derivative',pairs=4,code_derivative=gaps.grad[0,0].item(),doc_derivative=-1/(1+math.exp(-1)),loss=m.item())
 # Compare the exact fit_packs objective against independent NumPy-free scalars.
 vals={'b0':torch.tensor([[.25],[.75]]),'b1':torch.tensor([[.1],[.2]]),'b2':torch.tensor([[.0],[.5]]),'b3':torch.tensor([[.5],[.0]]),'st':torch.tensor([[.8,.1],[.2,.9]])}
 want=torch.tensor([[0.,0.,0.,0.],[1.,0.,0.,0.]]);member=torch.tensor([1.,0.])
 vals={k:v.requires_grad_() for k,v in vals.items()}
 class Opt:
  def zero_grad(self):pass
  def step(self):pass
 class Lat:
  _opt=Opt()
  def reset(self):pass
  def _drive(self,mat):return vals
 captured=[]
 original=torch.Tensor.backward
 def backward(t,*a,**k):captured.append(t.item());return original(t,*a,**k)
 with patch.object(torch.Tensor,'backward',backward):ltrain.fit_packs(Lat(),[(None,want,member)],1,.2)
 terms=[math.log1p(math.exp(.5)),(.1**2+.2**2)/2,(0**2+.5**2)/2,(.5**2+0**2)/2,math.log1p(math.exp(.4)),math.log1p(math.exp(.2))]
 exact=sum(terms)/6+.2*max(terms)
 emit('pack_formula',implementation=captured[0],independent=exact,difference=abs(captured[0]-exact))
 # Bootstrap reimplementation uses the declared percentile order statistic.
 import random
 diffs=[-.1,.2,.4];rng=random.Random(0);means=sorted(sum(rng.choices(diffs,k=3))/3 for _ in range(2000))
 # random.choices consumes random numbers differently than random.choice;
 # use independent randrange per draw for an exact seeded check.
 rng=random.Random(0);means=sorted(sum(diffs[rng.randrange(3)] for _ in range(3))/3 for _ in range(2000))
 got=promote.PromotionGate()._lower_bound(diffs)
 emit('bootstrap_formula',implementation=got,independent=means[100],index=100)
def feedback():
 found=False
 for seed in range(10):
  torch.manual_seed(100);fs=torch.rand(8,12).tolist()
  p=modules.ModularPolicy(seed=seed);q=modules.ModularPolicy(seed=seed)
  with torch.no_grad():
   for name,v in q._net.named_parameters():
    if 'fb_fc' in name:v.zero_()
  a=p.head_rates(fs,.5);b=q.head_rates(fs,.5)
  if a!=b:emit('intra_drive_feedback',seed=seed,head_rates_change=True);found=True;break
 if not found:emit('intra_drive_feedback',head_rates_change=False,seeds_tested=10)
 # On real recurrent loss, STOP can get gradient through feedback.
 p=modules.ModularPolicy(seed=0)
 g={'feats':torch.rand(12,12).tolist(),'pos':[0,1,2],'neg':list(range(3,12)),'ops':[0,1,2]*4,'gap':.5}
 p._group_loss(g).backward()
 emit('modular_indirect_stop_gradient',norm=p._net.st_fc.weight.grad[1].norm().item())
def sandbox_lineage():
 # Public command hook does not create an OS file-access boundary.
 with tempfile.TemporaryDirectory() as td:
  secret=Path(td)/'private.txt';secret.write_text('review sentinel')
  box=SandboxRunner(Path(td)/'case',cpu_s=1)
  r=box.run_command([sys.executable,'-c','import pathlib,sys;sys.exit(0 if pathlib.Path(sys.argv[1]).read_text()=="review sentinel" else 1)',str(secret)])
  emit('sandbox_host_access',status=r['status'],sentinel_outside_cwd=True)
  s=LineageStore(Path(td)/'lineage.db');s.record_intent('same','missing-mission',0,1,{'x':1});emit('lineage_unknown_mission',rows=len(s.commands_in('missing-mission','prepared')));s.close()
def regressions():
 import unittest
 from tests.test_rsi_evolve import RestoreOptimizerTests,StatelessScoringTests,PolicyLoopTests
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(C) for C in (RestoreOptimizerTests,StatelessScoringTests)])
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 emit('new_regressions',run=result.testsRun,failures=len(result.failures),errors=len(result.errors))
 # This individual test has no fixture dependencies; avoid its expensive class setup.
 PolicyLoopTests('test_snapshot_restore_turnover').test_snapshot_restore_turnover()
 emit('turnover_method',passed=True,class_setup='not invoked; method independent of fixture')
if __name__=='__main__':
 for n in sys.argv[1:] or ('packs','formula','feedback','sandbox_lineage','regressions'):globals()[n]()
