"""Independent positional measurement. No imports of prior audit scripts.
Fixed checkpoints: 5,40,50 epochs; seeds 0,1,2.
Rebuilds expression/value layers independently, composes value vectors,
computes exact random-order expectation, and validates held-out matches.
"""
import json, math, time, csv, sys, os
from pathlib import Path
from collections import defaultdict, Counter
from unittest.mock import patch
ROOT=Path(os.environ.get('REVIEW_REPO',Path.cwd())).resolve()
sys.path.insert(0,str(ROOT))
import torch
from rsi_programmer.policy import SearchPolicy, encode
from rsi_programmer import search, probe
OUT=Path(os.environ.get("REVIEW_OUTPUT",
        Path(__file__).resolve().parent / "measurement-rerun"))
torch.set_num_threads(1)
OPS=('add','sub','mul')
def calc(op,a,b):return tuple(x+y if op=='add' else x-y if op=='sub' else x*y for x,y in zip(a,b))
def features(op,l,r,target,ls,rs,size,bound,k,total):
 n=len(target);lo,hi=min(target),max(target)
 return [float(op==q) for q in OPS]+[ls/bound,rs/bound,size/bound]+[v for s in (l,r) for v in (sum(x==y for x,y in zip(s,target))/n,sum(lo<=x<=hi for x in s)/n)]+[k/total,1-k/total]
def build(task):
 # Each entry is (train vector, held vector, size). No repo replay,
 # enumerator, signature, layer builder, or feature encoder is used.
 rows=task['train'];held=task['held_out'];target=tuple(t[2] for t in rows);ht=tuple(t[2] for t in held)
 levels={0:[(tuple(r[c] for r in rows),tuple(r[c] for r in held),0) for c in (0,1)]};offset=0
 for size in range(1,task['max_tree_ops']+1):
  entries=[];seen=set();nextlevel=[]
  for ls in range(size):
   for l in levels[ls]:
    for r in levels[size-1-ls]:
     for op in OPS:
      v=calc(op,l[0],r[0]);h=calc(op,l[1],r[1]);entries.append((op,l,r,v,h))
      if (op,v) not in seen:seen.add((op,v));nextlevel.append((v,h,size))
  levels[size]=nextlevel
  solvers={i for i,e in enumerate(entries) if e[3]==target}
  if solvers:
   both={i for i in solvers if entries[i][4]==ht}
   return dict(task=task['id'],size=size,offset=offset,N=len(entries),solvers=solvers,both=both,entries=entries,target=target,bound=task['max_tree_ops'])
  offset+=len(entries)
 raise RuntimeError('no solution '+task['id'])
def fs(data,denom):
 total=10**9 if denom=='budget' else data['offset']+data['N']
 rows=[]
 for i,(op,l,r,_,__) in enumerate(data['entries']):
  f=features(op,l[0],r[0],data['target'],l[2],r[2],data['size'],data['bound'],data['offset']+i+1,total)
  assert f==encode(op,l[0],r[0],data['target'],l[2],r[2],data['size'],data['bound'],data['offset']+i+1,total)
  rows.append(f)
 return rows
def pos(scores,solvers):
 return next((k for k,i in enumerate(sorted(range(len(scores)),key=scores.__getitem__,reverse=True),1) if i in solvers),None)
def stream(task):
 rows=tuple(tuple(r[:2]) for r in task['train']);target=tuple(r[2] for r in task['train']);levels={0:list(probe.terminals())};count=0;recursive=0
 original=probe.signature
 def tracked(e,rs):
  nonlocal recursive
  recursive+=1;return original(e,rs)
 start=time.perf_counter()
 with patch.object(probe,'signature',new=tracked):
  for s in range(1,task['max_tree_ops']+1):
   seen=set();levels[s]=[]
   for ls in range(s):
    for l in levels[ls]:
     for r in levels[s-ls-1]:
      for op in OPS:
       e=probe.Expr(op,l,r,s);count+=1;v=probe.signature(e,rows)
       if v==target:return count,recursive,time.perf_counter()-start
       if (op,v) not in seen:seen.add((op,v));levels[s].append(e)
class Identity:
 def rank(self,feats,*args):return list(range(len(feats)))
def instrument(task):
 counts=Counter();orig=probe.signature
 def sp(e,rows):
  counts['recursive_total']+=1
  return orig(e,rows)
 def sg(e,rows):
  caller=sys._getframe(1).f_code.co_name
  counts['candidate' if caller=='_intern' else 'feature_operand']+=1
  return orig(e,rows)
 start=time.perf_counter()
 with patch.object(probe,'signature',new=sp),patch.object(search,'signature',new=sg):
  r=search.guided_search(Identity(),tuple(tuple(r[:2]) for r in task['train']),tuple(r[2] for r in task['train']),task['max_tree_ops'],10**9)
 wall=time.perf_counter()-start
 sc,sr,sw=stream(task)
 return dict(task=task['id'],guided_candidates=counts['candidate'],guided_operand_signatures=counts['feature_operand'],guided_recursive_including_operands=counts['recursive_total']+counts['candidate']+counts['feature_operand'],stream_candidates=sc,stream_recursive=sr,guided_tried=r['tried'],ratio=counts['candidate']/sc,guided_instrumented_wall=wall,stream_instrumented_wall=sw)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 families={k:json.load(open(ROOT/'fixtures'/f'rsi_{name}.json'))['tasks'] for k,name in [('shadow','shadow_tasks'),('novel','novel_tasks'),('ood_plain','ood_plain')]}
 built={k:[build(t) for t in ts] for k,ts in families.items()}
 # Measure training source overlap, exactly, before training.
 train_sigs={tuple(tuple(r) for r in t['train']) for t in families['shadow']}
 print('families',{k:len(v) for k,v in families.items()},'exact_task_train_overlap',{k:sum(tuple(tuple(r) for r in t['train']) in train_sigs for t in ts) for k,ts in families.items()},flush=True)
 groups=SearchPolicy().trace_groups(str(ROOT/'fixtures/rsi_shadow_tasks.json'))
 records=[]
 for seed in (0,1,2):
  p=SearchPolicy(seed=seed)
  for epoch in (0,5,40,50):
   if epoch:p.fit(groups,epochs=epoch-(0 if epoch==5 else 5 if epoch==40 else 40))
   for family,ds in built.items():
    for d in ds:
     for denom in ('budget','trace'):
      scores=p.scores(fs(d,denom));m=len(d['solvers']);mh=len(d['both'])
      records.append(dict(family=family,seed=seed,epoch=epoch,denom=denom,task=d['task'],N=d['N'],solver_count=m,held_solver_count=mh,probe=min(d['solvers'])+1,random_exact=(d['N']+1)/(m+1),position=pos(scores,d['solvers']),held_position=pos(scores,d['both']),random_held_exact=(d['N']+1)/(mh+1) if mh else None,offset=d['offset']))
   print('trained',seed,epoch,flush=True)
 with open(OUT/'positions.csv','w') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 summary=[]
 for family in families:
  for epoch in (0,5,40,50):
   for denom in ('budget','trace'):
    rs=[r for r in records if r['family']==family and r['epoch']==epoch and r['denom']==denom]
    seed_means=[sum(r['position'] for r in rs if r['seed']==s)/len(families[family]) for s in (0,1,2)]
    row=dict(family=family,epoch=epoch,denom=denom,mean=sum(seed_means)/3,seeds=seed_means,random_exact=sum(r['random_exact'] for r in rs)/len(rs),probe=sum(r['probe'] for r in rs)/len(rs),tasks_without_held_solver=sum(not d['both'] for d in built[family]))
    summary.append(row);print(json.dumps(row),flush=True)
 (OUT/'position-summary.json').write_text(json.dumps(summary,indent=2))
 work=[]
 for family,ts in families.items():
  for t in ts:
   r=instrument(t);r['family']=family;work.append(r);print('work',json.dumps(r),flush=True)
 (OUT/'signature-work.json').write_text(json.dumps(work,indent=2))
if __name__=='__main__':main()
