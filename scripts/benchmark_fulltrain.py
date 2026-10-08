"""Single-image synchronous benchmark: actual CNN and conditional actual VLM."""
import argparse, json, time, hashlib, platform, resource, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "medical_blood_pilot"))

def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
def rows(p):
    return {r['source_id']:r for r in map(json.loads, p.read_text().splitlines())}
def main():
    p=argparse.ArgumentParser()
    for key in ['project','root','base','adapter','probe','weights','output','cnn_reference','vlm_reference']:
        p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    p.add_argument('--mode',choices=['cnn','route','vlm'],required=True)
    p.add_argument('--policy',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    import torch, torchvision, sklearn
    from PIL import Image
    from run_wr50_frozen_test import load_backbone, restore_probe, predict, ManifestImages, evaluate
    from eval_fulltrain_test import LABELS, QUESTION, load_processor, MIN_PIXELS, IMAGE_SIZE, move_batch, model_inputs, set_seed, validate_test_records
    a.output.mkdir(exist_ok=False,parents=True)
    manifest=a.root/'bloodmnist_test_manifest.json'
    records,_=validate_test_records(json.loads(manifest.read_text())['records'])
    records=sorted(records,key=lambda r:r['source_id'])
    threshold=json.loads(a.policy.read_text())['main_recommendation']['threshold']
    set_seed(42); torch.set_num_threads(4)
    device=torch.device('cuda:0'); torch.cuda.reset_peak_memory_stats()
    t=time.perf_counter(); cnn=processor=model=None
    if a.mode!='vlm':
        scaler,head,_=restore_probe(a.probe); cnn=load_backbone(a.weights,device)
        transform=ManifestImages(records,a.root).transform
    cnn_load=time.perf_counter()-t
    t=time.perf_counter()
    if a.mode!='cnn':
        from peft import PeftModel
        from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
        processor=load_processor(a.base,MIN_PIXELS,IMAGE_SIZE)
        base=Qwen3_5ForConditionalGeneration.from_pretrained(a.base,torch_dtype=torch.bfloat16,trust_remote_code=True).to(device)
        model=PeftModel.from_pretrained(base,a.adapter).eval(); model.config.use_cache=False
    torch.cuda.synchronize(); vlm_load=time.perf_counter()-t
    def infer(record,force=False):
        torch.cuda.synchronize(); start=time.perf_counter()
        with Image.open(a.root/record['image']) as opened: image=opened.convert('RGB')
        read_ms=(time.perf_counter()-start)*1000
        confidence=None; cnn_pred=None; probs=None
        if cnn is not None:
            with torch.inference_mode(): feature=cnn(transform(image).unsqueeze(0).to(device)).cpu().numpy()
            pred,prob=predict(feature,scaler,head); cnn_pred=int(pred[0]); probs=prob[0].tolist(); confidence=float(prob[0].max())
        torch.cuda.synchronize(); cnn_ms=(time.perf_counter()-start)*1000-read_ms
        called=a.mode=='vlm' or (a.mode=='route' and (force or confidence<threshold))
        raw=None; vlm_ms=0.; final=cnn_pred
        if called:
            vt=time.perf_counter()
            prompt=processor.apply_chat_template([{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':QUESTION}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            batch=move_batch(processor(text=[prompt],images=[image],return_tensors='pt',min_pixels=MIN_PIXELS,max_pixels=IMAGE_SIZE))
            with torch.inference_mode(): generated=model.generate(**model_inputs(batch),max_new_tokens=32,do_sample=False)
            torch.cuda.synchronize()
            raw=processor.batch_decode(generated[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
            final=LABELS.index(raw) if raw in LABELS else -1
            vlm_ms=(time.perf_counter()-vt)*1000
        torch.cuda.synchronize()
        return dict(source_id=record['source_id'],label=record['label'],image=record['image'],prediction=final,cnn_prediction=cnn_pred,confidence=confidence,probabilities=probs,called=called,raw_answer=raw,read_ms=read_ms,cnn_ms=cnn_ms,vlm_ms=vlm_ms,e2e_ms=(time.perf_counter()-start)*1000)
    wt=time.perf_counter(); warm=[]
    for r in records[:5]: warm.append(infer(r,force=True))
    warm_seconds=time.perf_counter()-wt
    dump(a.output/'warmup.json',warm)
    torch.cuda.reset_peak_memory_stats(); measured=[]; loop=time.perf_counter()
    with (a.output/'predictions.jsonl').open('w') as f:
        for i,r in enumerate(records,1):
            row=infer(r); measured.append(row); f.write(json.dumps(row)+'\n'); f.flush()
            if i%100==0 or i==len(records): print(json.dumps(dict(event='progress',mode=a.mode,done=i,total=len(records))),flush=True)
    loop_seconds=time.perf_counter()-loop
    refcnn=rows(a.cnn_reference); refvlm=rows(a.vlm_reference)
    differences=[]
    for r in measured:
        c=refcnn[r['source_id']]; v=refvlm[r['source_id']]
        if r['label']!=c['label'] or r['label']!=v['label']: raise ValueError('label mismatch')
        expected_call=c['confidence']<threshold
        vp=LABELS.index(v['parsed']) if v['parsed'] in LABELS else -1
        diff={}
        if a.mode!='vlm':
            if r['cnn_prediction']!=c['prediction']: diff['cnn_prediction']=[c['prediction'],r['cnn_prediction']]
            if a.mode=='route' and r['called']!=expected_call: diff['route']=[expected_call,r['called']]
        if r['called'] and r['prediction']!=vp: diff['vlm_prediction']=[vp,r['prediction']]
        if diff: differences.append(dict(source_id=r['source_id'],differences=diff))
    lat=[r['e2e_ms'] for r in measured]
    report=dict(mode=a.mode,protocol=dict(batch_size=1,concurrency=1,prefetch=False,threshold=threshold,warmup_images=5,warmup_route_forces_vlm=True,timing='synchronized wall clock from disk image read through final prediction; excludes model loading, warmup and JSONL writes',order='source_id sorted',generation=dict(max_new_tokens=32,do_sample=False,enable_thinking=False,use_cache=False,min_pixels=MIN_PIXELS,max_pixels=IMAGE_SIZE)),environment=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,torchvision=torchvision.__version__,sklearn=sklearn.__version__,host=platform.node()),model_loading=dict(cnn_seconds=cnn_load,vlm_seconds=vlm_load),warmup_seconds=warm_seconds,metrics=evaluate(np.array([r['label'] for r in measured]),np.array([r['prediction'] for r in measured]),dict(enumerate(LABELS))),calls=sum(r['called'] for r in measured),latency=dict(mean_ms=float(np.mean(lat)),p95_ms=float(np.percentile(lat,95)),timed_throughput_images_s=len(lat)/(sum(lat)/1000),loop_wall_seconds=loop_seconds,loop_throughput_images_s=len(lat)/loop_seconds),resources=dict(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),parity=dict(difference_count=len(differences),cnn_max_probability_difference=max((abs(r['confidence']-refcnn[r['source_id']]['confidence']) for r in measured if r['confidence'] is not None),default=None)),provenance={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),manifest,a.probe,a.cnn_reference,a.vlm_reference,a.adapter/'adapter_model.safetensors']})
    dump(a.output/'differences.json',differences); dump(a.output/'report.json',report)
    print(json.dumps(report),flush=True)
if __name__=='__main__': main()
