#!/usr/bin/env python3
import argparse, json, tempfile
from pathlib import Path
import numpy as np, torch
from tools.train_human_behavior_embedding import Enc, normalize_local

def run(args):
    if args.smoke_test:
        d=Path(tempfile.mkdtemp())/'d'; d.mkdir(parents=True)
        traj=np.random.randn(40,80,4).astype(np.float32); np.save(d/'traj.npy',traj); data_dir=d
        ckpt=Path(args.checkpoint) if args.checkpoint else None
        if ckpt is None or not ckpt.exists():
            m=Enc(64); tmp=Path(tempfile.mkdtemp())/'m.pt'; torch.save({'model':m.state_dict(),'embedding_dim':64},tmp); ckpt=tmp
    else:
        data_dir=Path(args.data_dir); traj=np.load(data_dir/'traj.npy').astype(np.float32); ckpt=Path(args.checkpoint)
    obj=torch.load(ckpt,map_location='cpu'); ed=obj.get('embedding_dim',64); model=Enc(ed); model.load_state_dict(obj['model']); model.eval()
    out=[]; bs=args.batch_size
    with torch.no_grad():
        for i in range(0,len(traj),bs):
            x=torch.from_numpy(traj[i:i+bs]); z=model(normalize_local(x)); out.append(z.numpy())
    emb=np.concatenate(out,0)
    op=Path(args.out_path); op.parent.mkdir(parents=True,exist_ok=True); np.save(op,emb)
    (op.parent/'embedding_export_summary.json').write_text(json.dumps({'embedding_shape':list(emb.shape),'embedding_dim':int(emb.shape[1]),'dataset_path':str(data_dir),'checkpoint_path':str(ckpt)},indent=2))
    assert emb.shape[0]==len(traj)
    print('export_done')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data_dir',default='outputs/waymo_human_v1_full51'); p.add_argument('--checkpoint'); p.add_argument('--out_path',required=True)
    p.add_argument('--batch_size',type=int,default=1024); p.add_argument('--device',default='cpu'); p.add_argument('--smoke_test',action='store_true'); p.add_argument('--overwrite',action='store_true')
    run(p.parse_args())
