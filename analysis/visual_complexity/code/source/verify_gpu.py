"""Compare the original model on CUDA against completed TensorFlow CPU smoke results."""
import os
os.environ['KERAS_BACKEND']='torch'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from pathlib import Path
import json,time,argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference",type=Path,required=True,help="results.json from a TensorFlow CPU --smoke run")
args=parser.parse_args()
import numpy as np,cv2,py360convert,torch,keras
from PIL import Image
from keras.applications.inception_v3 import preprocess_input
root=Path(__file__).resolve().parents[2]
reference=json.loads(args.reference.read_text(encoding='utf-8'))
if reference.get('method',{}).get('backend','tensorflow')!='tensorflow':
    raise ValueError('Reference must use the TensorFlow backend')
if len(reference['faces'])!=6 or len(reference['scenes'])!=1:
    raise ValueError('Reference must contain exactly one scene and six faces')
assert torch.cuda.is_available()
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
print('GPU:',torch.cuda.get_device_name(0),flush=True)
model=keras.models.load_model(str(root/'code/models/trained_model_inception_v3.h5'),compile=False)
model.to('cuda')
print('Model device:',next(model.parameters()).device,flush=True)
p=root/'input'/reference['scenes'][0]['file']
a=cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_COLOR)
cube=py360convert.e2c(cv2.cvtColor(a,cv2.COLOR_BGR2RGB),512,mode='bilinear',cube_format='dict')
rows=[]
for r in reference['faces']:
    arr=cube[r['face']].astype(np.uint8)
    x=preprocess_input(np.asarray(Image.fromarray(arr).resize((299,299))).astype(np.float32))
    x=np.expand_dims(np.transpose(x,(2,0,1)),0)
    t=time.time(); y=float(np.squeeze(model.predict(x,verbose=0)))
    delta=y-r['DL']
    rows.append(dict(face=r['face'],tensorflow_cpu=r['DL'],torch_cuda=y,difference=delta,seconds=time.time()-t))
    print(rows[-1],flush=True)
result=dict(gpu=torch.cuda.get_device_name(0),rows=rows,max_abs_difference=max(abs(r['difference']) for r in rows),absolute_tolerance=1e-4)
result['passed']=result['max_abs_difference']<=result['absolute_tolerance']
(root/'code/gpu_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
assert result['passed'],result
print('GPU_EQUIVALENCE_PASSED',flush=True)
