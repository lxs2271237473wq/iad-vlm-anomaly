"""Read completed runs and report coverage; never present an incomplete macro as full AD2."""
import argparse,json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); tables=[]; completed=[]
    for marker in sorted(args.out.glob('*/*/complete.json')):
        info=json.loads(marker.read_text()); dataset,category=marker.parent.parts[-2:]
        completed.append(dict(dataset=dataset,category=category,**info))
        table=pd.read_csv(marker.parent/'metrics.csv'); table['dataset']=dataset; table['category']=category
        tables.append(table)
    if not tables: print('No completed categories'); return
    table=pd.concat(tables,ignore_index=True); table.to_csv(args.out/'completed_category_metrics.csv',index=False)
    macros=[]
    for dataset,expected in [('AD2',8),('VisA',12)]:
        part=table[table.dataset==dataset]; actual=part.category.nunique()
        if actual!=expected:
            print(dataset,'INCOMPLETE',actual,'/',expected,flush=True); continue
        for method,group in part.groupby('method'):
            assert len(group)==expected
            macros.append(dict(dataset=dataset,method=method,categories=expected,n=int(group.n.sum()),
                               macro_auroc=float(group.auroc.mean()),macro_ap=float(group.ap.mean())))
    pd.DataFrame(macros).to_csv(args.out/'complete_dataset_macro_metrics.csv',index=False)
    (args.out/'progress_snapshot.json').write_text(json.dumps(completed,indent=2))
    print(json.dumps(dict(completed=[f'{r["dataset"]}/{r["category"]}' for r in completed],macros=macros),indent=2))

if __name__=='__main__': main()
