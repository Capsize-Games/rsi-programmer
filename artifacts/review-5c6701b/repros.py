import sys, os, json, tempfile, math, copy, inspect
from pathlib import Path
from unittest.mock import patch
ROOT=Path(os.environ.get('REVIEW_REPO',Path.cwd())).resolve()
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
import torch
from rsi_programmer import probe, policy, modules, lattice, author, ltrain, lprecision
from rsi_programmer import promote, self_gate, autonomy, evolve, proxy, ood, tasks, loop_trainer, edits
from rsi_programmer.traces import TraceRecorder
from rsi_programmer.bytecode import binary_holds, run_vm, disassemble
from rsi_programmer.words import decode
from rsi_programmer.lineage import mission_ledger

torch.set_num_threads(1)
SH=str(ROOT/'fixtures/rsi_shadow_tasks.json')
FEATS=[[float(i==j) for j in range(12)] for i in range(6)]
ROWS=((1,1),(2,2),(3,3))
class Order:
 def scores(self,fs): return [0.0]*len(fs)
 def rank(self,fs,*args): return list(range(len(fs)))
def line(n,**data): print(str(n),json.dumps(data,sort_keys=True,default=str),flush=True)
def case51():
 for C in (policy.SearchPolicy,modules.ModularPolicy,lattice.BitLattice):
  p=C(seed=0,lr=.06); p.restore(p.snapshot())
  bound={id(t) for g in p._opt.param_groups for t in g['params']}=={id(t) for t in p._net.parameters()}
  before={k:v.clone() for k,v in p.weights().items()}
  if C is policy.SearchPolicy: p.fit([[([0.0]*12,1),([1.0]*12,0)]],epochs=2)
  elif C is modules.ModularPolicy:
   p.fit([{'feats':FEATS,'pos':[0,1,2],'neg':[3,4,5],'ops':[0,1,2,0,1,2],'gap':1.0}],epochs=2)
  else: ltrain.fit_packs(p,[ltrain.pack_rows([(f,i%2) for i,f in enumerate(FEATS)])],2,.2)
  line(51,network=C.__name__,bound=bound,moved=sum(not torch.equal(v,before[k]) for k,v in p.weights().items()),lr=p._opt.param_groups[0]['lr'])
def case52():
 for C in (policy.SearchPolicy,modules.ModularPolicy,lattice.BitLattice):
  p=C(seed=0)
  score=p.scores if hasattr(p,'scores') else p.head_rates
  a=score(FEATS); b=score(FEATS); score([[.5]*12]*6); c=score(FEATS)
  line(52,network=C.__name__,repeat_equal=a==b,history_equal=a==c)
 p=modules.ModularPolicy(seed=0); a=p.head_rates([[.5]*12]*4,.9)
 q=modules.ModularPolicy(seed=0)
 with torch.no_grad():
  for name,t in q._net.named_parameters():
   if 'fb_fc' in name:t.zero_()
 b=q.head_rates([[.5]*12]*4,.9)
 line(52,within_drive_feedback_changes_outputs=a!=b)
def case53():
 rows=((1,2),(2,5)); target=(3,7)
 e,t,_=probe.enumerate_programs(rows,target,1)
 from rsi_programmer.search import guided_search
 out=guided_search(Order(),rows,target,1,1)
 line(53,enumerator_tried=t,guided_tried=out['tried'],solved=out['solved'],budget=1,nodes=len(out['graph']['nodes']))
def case54():
 from rsi_programmer.search import guided_search
 e,_,_=probe.enumerate_programs(ROWS,(2,4,6),1)
 oracle=lambda g: probe.interpret(g,1,2)==3
 g=guided_search(Order(),ROWS,(2,4,6),1,100,oracle=oracle)
 h=guided_search(Order(),ROWS,(3,6,9),2,10000,oracle=lambda g:probe.interpret(g,1,2)==5)
 line(54,enum_held=probe.interpret(probe.lower_to_dag(e),1,2),guided_with_oracle_recovers=g['solved'],lower_layer_collision_solved=h['solved'],required='x+2*y within bound 2')
def case55():
 triples=policy.layer_triples({0:list(probe.terminals())},1)
 tr=policy._label_final(triples,ROWS,(2,4,6),1,1,0,12)
 inf=policy._label_final(triples,ROWS,(2,4,6),1,1,0,10**9)
 mat,_,_=ltrain.pack_rows([(f,y) for f,y in tr])
 p=policy.SearchPolicy(seed=0); p._bonus=lambda fs:[1000*i for i in range(len(fs))]
 q=policy.SearchPolicy(seed=0); q.restore(p.snapshot())
 line(55,train_frac_last=tr[-1][0][-2:],infer_frac_last=inf[-1][0][-2:],training_gap=sorted(set(mat[:,-1].tolist())),bonus_roundtrip_equal=p.rank(FEATS)==q.rank(FEATS),author_counter='constant in _score_order; increments in _label_layer')
def case56():
 out=torch.tensor([1.,0.],requires_grad=True)
 loss=ltrain._margins(out,torch.tensor([1.,0.])); loss.backward()
 line(56,perfect_margin_loss=loss.item(),rate_gradient=out.grad.tolist(),atan_derivative_at_10=1/(1+(math.pi*10)**2),max_over_terms='six scalar aggregate losses, including two struct heads')
 p=lattice.BitLattice(seed=0); pack=ltrain.pack_rows([(f,i%2) for i,f in enumerate(FEATS)])
 ltrain.fit_packs(p,[pack],1,.2)
 line(56,stop_weight_grad_norm=p._net.st_fc.weight.grad[1].norm().item())
def case57():
 for seed in range(3):
  lat=lattice.BitLattice(seed=seed)
  torch.manual_seed(77); mat=torch.rand(60,13)
  found=ltrain._calibrate(lat,mat)
  after={k:ltrain._head_rate(lat,mat,k,0) for k in found}
  line(57,seed=seed,reported=found,final=after,max_drift=max(abs(found[k]-after[k]) for k in found))
def case58():
 groups=[[(FEATS[i],i%2) for i in range(6)]]; seen={}
 def fit(lat,packs,*a): seen['fit']={tuple(r) for m,w,b in packs for r in m[:,:12].tolist()}
 def refine(lat,gs):
  _,val=lprecision.split_groups(gs)
  seen['val']={tuple(f) for f,y in val}; return {}
 with tempfile.TemporaryDirectory() as td, patch.object(ltrain,'contrastive_groups',return_value=groups),patch.object(ltrain,'_calibrate',return_value={}),patch.object(ltrain,'fit_packs',side_effect=fit),patch.object(lprecision,'refine',side_effect=refine),patch.object(ltrain,'_report',return_value={}):
  ltrain.train_lattice(lattice.BitLattice(),'',td+'/a.db',epochs=1)
 line(58,validation_rows=len(seen['val']),also_fitted=len(seen['val']&seen['fit']))
def case59():
 g=promote.PromotionGate()
 line(59,singleton=g.decide([0.],[.1],0,0,0),iid_lower_two=g._lower_bound([0.,1.]),iid_lower_repeated=g._lower_bound([0.,1.]*50))
 with tempfile.TemporaryDirectory() as td:
  gate=self_gate.SelfGate(td,td+'/a.db'); calls=[]
  def decide(*a): calls.append(a[2:]); return {'promote':False,'reasons':['test']}
  with patch.object(gate._gate,'decide',side_effect=decide): gate.propose(edits.DEFAULT_CONFIG,{'op':'set','field':'lr','value':.1},lambda *a:{'held':[1.]})
  gate.close(); line(59,self_gate_safety_tokens=calls[0][:2])
def case60():
 line(60,singletons=autonomy.autonomy_ratio({'a':[0.],'b':[1.]}),variable_promoted=autonomy.autonomy_ratio({'promoted':[.05,100.],'refused':[0.,0.]}),truncated_distance=autonomy._dist([0.],[0.,100.]),profile=autonomy.agency_profile([{'generation':i,'weights':[i],'fitness':{'correctness':i}} for i in range(3)]))
def case61():
 class Config(dict):
  def get(self,k,*a): used.add(k); return super().get(k,*a)
 used=set()
 class P:
  def __init__(self,*a,**k):pass
  def trace_groups(self,*a):return []
  def fit(self,*a):pass
 with patch.object(loop_trainer,'SearchPolicy',P),patch.object(loop_trainer,'refine_operands'),patch.object(loop_trainer,'_solves',return_value=[]):loop_trainer.train_scores(Config(edits.DEFAULT_CONFIG),0,'')
 unused=set(edits.FIELDS)-used
 line(61,consumed=sorted(used),unused=sorted(unused),inert_edits=sum(e['field'] in unused for e in edits.all_edits(edits.DEFAULT_CONFIG)),total_edits=len(edits.all_edits(edits.DEFAULT_CONFIG)),documented='inert' in loop_trainer.__doc__)
def case62():
 obj=evolve.PolicyLoop.__new__(evolve.PolicyLoop)
 class Screen:
  def rank_policies(self,ps):return {'good':1,'bad':0}
  def fidelity(self,*a):return {}
 obj._screen=Screen(); obj._stop_after=1
 fit={'gates_pass':True,'correctness':1.,'efficiency':1.,'recovery':1.}
 obj.evaluate=lambda p:{**fit,'efficiency':.5}
 obj._persist=lambda *a:None
 pop=[{'id':n,'parent':'p','policy':n} for n in ('good','bad')]
 obj._evaluate_pop(pop,[{'id':'p','fitness':fit}],1)
 line(62,entries=pop,selected=max(pop,key=evolve._fitkey)['id'])
def case63():
 obj=evolve.PolicyLoop.__new__(evolve.PolicyLoop); obj._budget=10000; obj._tasks=[0,1];obj._probe=lambda p:0.
 obj._score_task=lambda p,t:{'solved':t==0,'cost':10,'gate':'' if t==0 else 'miss','recovery':1.}
 one=obj.evaluate(None)
 obj._score_task=lambda p,t:{'solved':False,'cost':10,'gate':'miss','recovery':0.}
 none=obj.evaluate(None)
 obj._seed=0;obj._rows=lambda *a:(ROWS,(2,4,6),1)
 with patch.object(evolve,'guided_search',return_value={'solved':True,'tried':200}):r=obj._recovery(policy.SearchPolicy(),{'id':'t'},100)
 line(63,half_vs_none_identical=one==none,recovery_for_2x_cost=r,gating_documented='Gated' in evolve.PolicyLoop.evaluate.__doc__)
def case64():
 task={'id':'t','train':[[1,2,3]],'held_out':[[2,3,5]],'max_tree_ops':1}; seen={}
 def ref(pol,rs,**kw):seen['refine']=[r[0] for r in rs];return [0,0,0,1],{}
 def beam(*a,**k):seen.setdefault('eval',[]).append(a[5]);return {'solved':False,'tried':1}
 with patch.object(ood,'refine_author',side_effect=ref):w=ood._refine_locked(policy.SearchPolicy(),[task])
 with patch.object(ood,'beam_search',side_effect=beam):ood.evaluate(policy.SearchPolicy(),[task],TraceRecorder(SH),10,weights=w)
 p=policy.SearchPolicy();before=ood.weights_hash(p); p._bonus=lambda fs:[100.]*len(fs)
 line(64,refined_ids=seen['refine'],evaluated_ids=seen['eval'],behavior_change_covered_by_hash=before!=ood.weights_hash(p),held_rows_in_refine=False)
def case65():
 p=modules.ModularPolicy(); out={'op':torch.tensor([[.9,.95,.95],[.9,.1,.9]],requires_grad=True),'lo':torch.tensor([[.9],[.1]],requires_grad=True),'ro':torch.tensor([[.9],[.1]],requires_grad=True),'st':torch.tensor([[.9,.95],[.1,.95]],requires_grad=True)}
 p._drive=lambda m:out
 g={'feats':FEATS[:2],'pos':[0],'neg':[1],'ops':[0,1],'gap':1.}
 p._group_loss(g).backward()
 line(65,op_argmax=out['op'].argmax(dim=1).tolist(),op_truth=g['ops'],op_grad=out['op'].grad.tolist(),direct_stop_grad=out['st'].grad[:,1].tolist(),caveat='STOP can still affect encoder through delayed feedback')
def case66():
 x,y=probe.terminals();rows=((1,2),(2,5),(3,1));target=tuple(a-b for a,b in rows)
 ts=policy.layer_triples({0:[x,y]},1);want=author.solution_sigs(rows,target,3)
 g=author._label_layer(1,ts,rows,target,3,0,100,want)
 zero=probe.Expr('sub',x,x,1); alt=probe.Expr('add',x,probe.Expr('sub',zero,y,2),3)
 idx=ts.index(('sub',x,x))
 line(66,valid_alternative=probe.signature(alt,rows)==target,alternative_intermediate_label=g[idx][1],canonical_signature_count=len(want))
def case67():
 graph=decode([0,999,-3,8,4242]); catches={}
 for name,code in [('store',b'\x05\x00\x00'),('add',b'\x06\x00'),('empty',b'\x00')]:
  try:catches[name]=binary_holds({'train':[[1,2,0]],'held_out':[]},code)
  except Exception as e:catches[name]=type(e).__name__+': '+str(e)
 bg={'input_types':{'x':'Int','y':'Int'},'nodes':[{'op':'lt','type':'Bool','args':['x','y']}],'output':0}
 probe.validate_graph(bg)
 line(67,wrapped_graph=graph,binary_holds=catches,bool_validated=True,undefined_fetch=run_vm(b'\x04\x00\x00',1,2),early_halt=disassemble(b'\x01\x00\x02\x00'))
def case68():
 s=proxy.ProxyScreen(SH,str(ROOT/'fixtures/rsi_novel_tasks.json'),seeds=(10,20,30));s._groups=lambda:[];seen=[]
 with patch.object(proxy,'_trained',side_effect=lambda seed,*a:seen.append(seed)),patch.object(proxy,'_mean_gain',return_value=0.):rep=s.screen([{'id':'a','seed':100}])
 line(68,configured=[10,20,30],observed=seen,reported=rep['seeds'])
def case69():
 with tempfile.TemporaryDirectory() as td:
  g=self_gate.SelfGate(td,td+'/a.db');g._record('first',{'x':1});d2=g._record('second',{'x':2});g.close()
  g=self_gate.SelfGate(td,td+'/a.db');d=g._record('third',{'x':3});rs=g._log.commands_in('self-gate','prepared');g.close()
  line(69,rows=len(rs),kinds=[json.loads(r['intent'])['kind'] for r in rs],returned_second_digest=d==d2,returned_written_digest=d==rs[0]['digest'])
def case70():
 out=author.author_search(Order(),((1,2),(2,5)),(3,7),1,100,topk=1,weights=[0,0,1000,0],fallback=False)
 line(70,zero_snn_exact_weight_solves=out['solved'],nn_features=policy.DIM,extra_features=len(author.EXTRA),gap_before_commit=author.open_gap([], (3,7)),gap_with_solver=author.open_gap([(3,7)],(3,7)))
def case71():
 x,y=probe.terminals();a=probe.Expr('add',x,y,1)
 line(71,recount=tasks._size_of('add',a,a,None),true_size=3)

def extras():
 from rsi_programmer.search import guided_search
 from rsi_programmer.sandbox import SandboxRunner
 # NaN bypass.
 line('E-nan',verdict=promote.PromotionGate().decide([0.],[float('nan')],0,0,0))
 # Default trainer credits train fit which fails held rows.
 with tempfile.TemporaryDirectory() as td:
  task={'id':'collision','train':[[1,1,2],[2,2,4],[3,3,6]],'held_out':[[1,2,3]],'max_tree_ops':1}
  path=Path(td)/'family.json';path.write_text(json.dumps({'tasks':[task]}))
  out=guided_search(Order(),ROWS,(2,4,6),1,10**9)
  score=loop_trainer._solves(Order(),str(path))
  line('E-oracle',score=score,oracle_holds=TraceRecorder(str(path)).oracle_holds(task,out['graph']))
 # Validator malformed opcode escapes prescreen as TypeError.
 g={'input_types':{'x':'Int','y':'Int'},'nodes':[{'op':[],'type':'Int','args':['x','y']}],'output':0}
 with tempfile.TemporaryDirectory() as td:
  try: r=SandboxRunner(Path(td)).run_candidate(g,[[1,2]])
  except Exception as e:r=type(e).__name__+': '+str(e)
  line('E-validation',outcome=r)
 # Freeze state lost on restore.
 p=lattice.BitLattice();p.set_encoder_frozen(True)
 before=sum(not t.requires_grad for t in p._net.parameters());p.restore(p.snapshot());after=sum(not t.requires_grad for t in p._net.parameters())
 line('E-freeze',frozen_before=before,frozen_after=after)
 # Same-store cross-thread access despite a mutex.
 import concurrent.futures
 with tempfile.TemporaryDirectory() as td:
  s=mission_ledger(td+'/a.db','m')
  with concurrent.futures.ThreadPoolExecutor(1) as ex:
   try:r=ex.submit(s.commands_in,'m','prepared').result()
   except Exception as e:r=type(e).__name__+': '+str(e).split('.')[0]
  s.close();line('E-thread',outcome=r)
 # Validation loss is invariant to all cross-bit per-example alignment.
 # Adam continuation / option loss.
 p=policy.SearchPolicy(lr=.07); group=[[([0.]*12,1),([1.]*12,0)]];p.fit(group,epochs=2)
 state=len(p._opt.state);p._opt.param_groups[0]['weight_decay']=.1;p._opt.param_groups[0]['betas']=(.8,.9)
 p.restore(p.snapshot());line('E-adam',state_before=state,state_after=len(p._opt.state),lr=p._opt.param_groups[0]['lr'],weight_decay=p._opt.param_groups[0]['weight_decay'],betas=p._opt.param_groups[0]['betas'])

if __name__=='__main__':
 names=sys.argv[1:] or [str(i) for i in range(51,72)]
 for n in names:
  try: extras() if n=='extras' else globals()['case'+n]()
  except Exception as e:
   import traceback; traceback.print_exc(); line(n,REPRO_ERROR=repr(e))
