#!/usr/bin/env python3
"""Frozen BloodMNIST eight-class Qwen baseline/LoRA pilot; no test input."""
import argparse, json, random, time
from collections import Counter
from pathlib import Path
from vlm_common import (IMAGE_SIZE,MIN_PIXELS,build_supervised_batch,load_image,load_processor,move_batch,model_inputs,set_seed,setup_lora,write_json)

LABELS=("basophil","eosinophil","erythroblast","immature_granulocytes","lymphocyte","monocyte","neutrophil","platelet")
QUESTION="Which blood cell type is shown? Answer with exactly one of: "+"; ".join(LABELS)+"."
def load(path):
 x=json.load(path.open()); return x["records"] if isinstance(x,dict) and isinstance(x.get("records"),list) else (_ for _ in ()).throw(ValueError("expected top-level records"))
def main():
 p=argparse.ArgumentParser(); p.add_argument("--train-json",type=Path,required=True);p.add_argument("--dev-json",type=Path,required=True);p.add_argument("--image-root",type=Path,required=True);p.add_argument("--base-model",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 for split,path,limit in (("train",a.train_json,250),("dev",a.dev_json,50)):
  rows=load(path); ids=set(); counts=[0]*8
  for r in rows:
   if set(("image","source_id","label","class_name","split"))-set(r) or r["split"]!=split or not isinstance(r["label"],int) or not 0<=r["label"]<8 or r["class_name"]!=LABELS[r["label"]] or r["source_id"] in ids:raise ValueError("invalid BloodMNIST record")
   ids.add(r["source_id"]);counts[r["label"]]+=1
  if any(c<=0 or c>limit for c in counts):raise ValueError("invalid per-class count")
  if any(r.get("official_split") != ("train" if split=="train" else "val") for r in rows):raise ValueError("official split mismatch")
  for r in rows:r.setdefault("question",QUESTION);r.setdefault("answer",r["class_name"]);r.setdefault("task","blood_cell_classification");r.setdefault("template_id","blood8")
  if split=="train":train=rows
  else:dev=rows
 import torch
 from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
 if {r["source_id"] for r in train}&{r["source_id"] for r in dev} or {r["image"] for r in train}&{r["image"] for r in dev}:raise ValueError("train/dev overlap")
 started=time.perf_counter();set_seed(42);a.output_dir.mkdir(parents=True);proc=load_processor(a.base_model,MIN_PIXELS,IMAGE_SIZE)
 def score(model,name):
  out=[]; model.eval()
  from PIL import Image
  for r in dev:
   im=load_image(a.image_root,r); prompt=proc.apply_chat_template([{"role":"user","content":[{"type":"image","image":im},{"type":"text","text":QUESTION}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False); b=move_batch(proc(text=[prompt],images=[im],return_tensors="pt",min_pixels=MIN_PIXELS,max_pixels=IMAGE_SIZE),"cuda")
   with torch.inference_mode(): g=model.generate(**model_inputs(b),max_new_tokens=32,do_sample=False)
   raw=proc.batch_decode(g[:,b["input_ids"].shape[1]:],skip_special_tokens=True)[0].strip(); parsed=raw if raw in LABELS else "__invalid__";out.append({"source_id":r["source_id"],"gt":r["answer"],"raw_answer":raw,"parsed":parsed,"label":r["label"]})
  cm=[[0]*9 for _ in range(8)]
  for x in out:cm[x["label"]][LABELS.index(x["parsed"]) if x["parsed"] in LABELS else 8]+=1
  recalls=[cm[i][i]/sum(cm[i]) for i in range(8)];f1=[]
  for i in range(8):
   tp=cm[i][i]; fp=sum(cm[j][i] for j in range(8) if j!=i); fn=sum(cm[i][j] for j in range(9) if j!=i); f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
  m={"accuracy":sum(cm[i][i] for i in range(8))/len(out),"macro_f1":sum(f1)/8,"per_class_recall":recalls,"confusion_8x9":cm,"invalid":sum(x["parsed"]=="__invalid__" for x in out)};write_json(a.output_dir/(name+"_metrics.json"),m);open(a.output_dir/(name+"_predictions.jsonl"),"w").write("".join(json.dumps(x)+"\n" for x in out));return m
 base=Qwen3_5ForConditionalGeneration.from_pretrained(a.base_model,torch_dtype=torch.bfloat16,trust_remote_code=True).to("cuda");score(base,"baseline")
 model,_,_,trainable,expected=setup_lora(base);model.config.use_cache=False
 if sum(trainable.values())!=expected:raise RuntimeError("LoRA shape guard failed")
 opt=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=2e-4,weight_decay=.01);best=None
 for e in (1,2):
  model.train();rs=list(train);random.Random(42+e).shuffle(rs)
  for r in rs:
   b,_=build_supervised_batch(r,load_image(a.image_root,r),proc,type("A",(),{"min_pixels":MIN_PIXELS,"max_pixels":IMAGE_SIZE})());o=model(**move_batch(b,"cuda"));o.loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();opt.zero_grad()
  model.save_pretrained(a.output_dir/f"epoch_{e:02d}");m=score(model,f"epoch_{e:02d}");best=max(best or (m["macro_f1"],e),(m["macro_f1"],e))
 write_json(a.output_dir/"best_checkpoint.json",{"epoch":best[1],"macro_f1":best[0]});write_json(a.output_dir/"run_metadata.json",{"train_json":str(a.train_json),"dev_json":str(a.dev_json),"base_model":str(a.base_model),"trainable":sum(trainable.values()),"lr":2e-4,"steps":4000,"prompt":QUESTION,"seed":42,"elapsed_seconds":time.perf_counter()-started})
if __name__=="__main__":main()
