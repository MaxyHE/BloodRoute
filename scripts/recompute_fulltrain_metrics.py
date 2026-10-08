"""Recompute the new serial benchmark from public per-image records; standard library only."""
import json, math, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/fulltrain-20261005'

def percentile(values,p):
    values=sorted(values);position=(len(values)-1)*p
    lo=math.floor(position);hi=math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)

def compute(rows):
    confusion=[[0]*9 for _ in range(8)]
    for r in rows:
        label=r['label']; pred=r['prediction']
        assert label in range(8)
        confusion[label][pred if pred in range(8) else 8]+=1
    f1=[]
    for c in range(8):
        tp=confusion[c][c];fp=sum(confusion[i][c] for i in range(8) if i!=c)
        fn=sum(confusion[c][i] for i in range(9) if i!=c)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return {'images':len(rows),'correct':sum(confusion[c][c] for c in range(8)),
            'accuracy':sum(confusion[c][c] for c in range(8))/len(rows),
            'macro_f1':statistics.mean(f1),'calls':sum(r['called'] for r in rows),
            'mean_ms':statistics.mean(r['e2e_ms'] for r in rows),
            'p95_ms':percentile([r['e2e_ms'] for r in rows],.95)}

def main():
    expected=json.loads((DATA/'independent-verification.json').read_text())
    records={};results={}
    for mode in ('cnn','vlm','route'):
        rows=[json.loads(line) for line in (DATA/f'{mode}-predictions.jsonl').read_text().splitlines()]
        assert len(rows)==len({r['source_id'] for r in rows})==3421
        records[mode]={r['source_id']:r for r in rows}
        results[mode]=compute(rows)
        for key in results[mode]:
            assert math.isclose(results[mode][key],expected[mode][key],rel_tol=1e-10,abs_tol=1e-8),(mode,key)
        print(mode,json.dumps(results[mode]))
    assert all(set(records[m])==set(records['cnn']) for m in records)
    corrected=harmed=0
    for sid,r in records['route'].items():
        c=records['cnn'][sid];v=records['vlm'][sid]
        assert r['label']==c['label']==v['label']
        assert r['prediction']==(v['prediction'] if r['called'] else c['prediction'])
        corrected+=c['prediction']!=r['label'] and r['prediction']==r['label']
        harmed+=c['prediction']==r['label'] and r['prediction']!=r['label']
    assert (corrected,harmed)==(164,33)
    reduction=1-results['route']['mean_ms']/results['vlm']['mean_ms']
    print(f'Paired: corrected={corrected}, harmed={harmed}; mean latency reduction={reduction:.2%}')

if __name__=='__main__':
    main()
