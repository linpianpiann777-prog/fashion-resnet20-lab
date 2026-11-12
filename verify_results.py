"""Reload every checkpoint and independently verify the saved final test evidence."""
import csv
import json
import hashlib
import numpy as np
import torch
from models import ResNet20,complexity
from data_utils import ROOT,load_part,tensor_images
from train import evaluate

def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images,labels=load_part(False);x=tensor_images(images,device);y=torch.as_tensor(labels.astype(np.int64),device=device)
    summary=json.loads((ROOT/'results/summary.json').read_text())
    checks=[]
    for r in summary:
        folder=ROOT/f'results/{r["variant"]}_seed{r["seed"]}'
        checkpoint=torch.load(folder/'best.pt',map_location=device,weights_only=True)
        model=ResNet20(r['variant']).to(device);model.load_state_dict(checkpoint['state_dict'])
        stats=complexity(model)
        assert all(stats[k]==r[k] for k in stats)
        loss,accuracy,probs=evaluate(model,x,y)
        stored=np.load(folder/'test_predictions.npz')
        assert np.array_equal(stored['labels'],labels)
        assert np.allclose(probs,stored['probabilities'],atol=3e-5)
        predictions=probs.argmax(1)
        assert int((predictions==labels).sum())==r['test_correct']
        cm=np.bincount(labels.astype(np.int64)*10+predictions,minlength=100).reshape(10,10)
        precision=np.diag(cm)/np.maximum(cm.sum(0),1);recall=np.diag(cm)/cm.sum(1)
        f1=2*precision*recall/np.maximum(precision+recall,1e-12)
        assert abs(f1.mean()-r['macro_f1'])<1e-12
        with (folder/'history.csv').open() as f:history=list(csv.DictReader(f))
        assert len(history)==20
        best=max(history,key=lambda row:float(row['validation_accuracy']))
        assert int(best['epoch'])==r['best_epoch']==checkpoint['epoch']
        checks.append({'variant':r['variant'],'seed':r['seed'],'test_correct':int((predictions==labels).sum()),
                       'max_probability_absolute_difference':float(np.max(np.abs(probs-stored['probabilities']))),
                       'checkpoint_sha256':hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()})
        print('Verified',r['variant'],r['seed'],r['test_correct'],flush=True)
    (ROOT/'results/reproducibility_checks.json').write_text(json.dumps({'passed':True,'device':str(device),'checks':checks},indent=2))

if __name__=='__main__':main()
