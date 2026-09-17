"""Recompute six panorama scenes with the original FC/DL algorithms."""
from pathlib import Path
import argparse, csv, json, os, re, time
from datetime import datetime
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[2]
def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'input')
    ap.add_argument('--model',type=Path,default=ROOT/'code/models/trained_model_inception_v3.h5')
    ap.add_argument('--output',type=Path,default=ROOT/'output')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--backend',choices=['torch','tensorflow'],default='torch')
    args=ap.parse_args()
    files={}
    for p in args.input.iterdir():
        if p.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}: continue
        m=re.search(r'\b(C[01])\s+W(15|45|75)\b',p.stem)
        if not m: raise ValueError(f'Unrecognized condition: {p.name}')
        k=(int(m[2]),m[1])
        if k in files: raise ValueError(f'Duplicate condition: {k}')
        if k[1]=='C1' and not re.search(r'\bLA\b',p.stem): raise ValueError(f'Expected LA: {p.name}')
        files[k]=p
    if set(files)!={(w,c) for w in (15,45,75) for c in ('C0','C1')}: raise ValueError('Expected exactly six conditions')
    with args.model.open('rb') as f:
        if f.read(8)!=b'\x89HDF\r\n\x1a\n': raise ValueError('Model must be HDF5, not a Git LFS pointer')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    os.environ.setdefault('MPLBACKEND','Agg')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ['KERAS_BACKEND']=args.backend
    if args.backend=='tensorflow':
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    else:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU unavailable; use --backend tensorflow for CPU')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        print('GPU:',torch.cuda.get_device_name(0),flush=True)
    import keras
    import numpy as np
    import cv2
    from PIL import Image
    import py360convert
    from visual_clutter import Vlc
    from keras.applications.inception_v3 import preprocess_input
    print('Loading original model',flush=True)
    model=keras.models.load_model(str(args.model),compile=False)
    if args.backend=='torch':
        model.to('cuda')
        print('Model device:',next(model.parameters()).device,flush=True)
    print('Model input:',model.input_shape,'output:',model.output_shape,flush=True)
    if model.input_shape[1]!=3 or any(d not in (None,299) for d in model.input_shape[2:]): raise ValueError('Unexpected model shape')
    run=args.output/(('smoke_' if args.smoke else 'run_')+datetime.now().strftime('%Y%m%d_%H%M%S'))
    run.mkdir(parents=True,exist_ok=False)
    tmp=run/'working'; tmp.mkdir()
    print('RUN_DIR:',run,flush=True)
    keys=[(45,'C0')] if args.smoke else sorted(files)
    faces_out=[]; scenes=[]; start=time.time()
    for wwr,condition in keys:
        p=files[(wwr,condition)]
        pano=cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_COLOR)
        if pano is None or pano.shape[1]!=2*pano.shape[0]: raise ValueError(f'Invalid 2:1 panorama: {p.name}')
        rgb=cv2.cvtColor(pano,cv2.COLOR_BGR2RGB)
        cube=py360convert.e2c(rgb,512,mode='bilinear',cube_format='dict')
        if len(cube)!=6: raise ValueError('Expected six cube faces')
        current=[]
        for face,arr in cube.items():
            face_started=time.time()
            arr=arr.astype(np.uint8)
            v=Vlc(arr,output_dir=str(tmp),prefix='face')
            raw=v.getClutter_FC()
            fc=float(raw[0]) if isinstance(raw,tuple) else float(raw)
            fc_seconds=time.time()-face_started
            img=Image.fromarray(arr).resize((299,299))
            x=preprocess_input(np.asarray(img).astype(np.float32))
            x=np.expand_dims(np.transpose(x,(2,0,1)),0)
            dl=float(np.squeeze(model.predict(x,verbose=0)))
            if not np.isfinite([fc,dl]).all(): raise ValueError('Non-finite score')
            row=dict(file=p.name,WWR=wwr,condition=condition,face=face,FC=fc,DL=dl)
            current.append(row); faces_out.append(row)
            write_csv(run/'face_scores.csv',faces_out)
            print(f'{condition} W{wwr} {face}: FC={fc:.9f} DL={dl:.9f} FC_seconds={fc_seconds:.2f} total_seconds={time.time()-face_started:.2f}',flush=True)
        scenes.append(dict(file=p.name,WWR=wwr,condition=condition,FC=float(np.mean([r['FC'] for r in current])),DL=float(np.mean([r['DL'] for r in current]))))
    pairs=[]
    if not args.smoke:
        for metric in ('FC','DL'):
            vals=np.array([s[metric] for s in scenes])
            if vals.std(ddof=0)==0: raise ValueError(f'Zero variance: {metric}')
            z=(vals-vals.mean())/vals.std(ddof=0)
            assert abs(z.mean())<1e-10 and abs(z.std(ddof=0)-1)<1e-10
            for s,v in zip(scenes,z): s['z_'+metric]=float(v)
        for s in scenes: s['Score']=0.5*s['z_FC']+0.5*s['z_DL']
        for w in (15,45,75):
            a,b=[s for s in scenes if s['WWR']==w]
            pairs.append(dict(WWR=w,FC_C0=a['FC'],FC_C1=b['FC'],delta_FC=b['FC']-a['FC'],DL_C0=a['DL'],DL_C1=b['DL'],delta_DL=b['DL']-a['DL']))
        write_csv(run/'pair_comparisons.csv',pairs)
    write_csv(run/'scene_scores.csv',scenes)
    packages={p:version(p) for p in ['tensorflow','keras','numpy','pillow','opencv-python-headless','py360convert','pyrtools','visual-clutter']}
    if args.backend=='torch': packages['torch']=version('torch')
    result=dict(scenes=scenes,faces=faces_out,pairs=pairs,packages=packages,model=args.model.name,elapsed_seconds=time.time()-start,method=dict(cube_face_width=512,interpolation='bilinear',DL_input=[3,299,299],aggregation='arithmetic mean of six faces',z_reference='current six scenes',ddof=0,weights=dict(FC=0.5,DL=0.5)))
    result['method']['backend']=args.backend
    result['method']['device']=torch.cuda.get_device_name(0) if args.backend=='torch' else 'CPU'
    (run/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(scenes,ensure_ascii=False,indent=2),flush=True)
    print('COMPLETE:',run,flush=True)
if __name__=='__main__': main()
