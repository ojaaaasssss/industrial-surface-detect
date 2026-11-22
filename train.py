import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from src.core import DefectCNN, transform, LABELS

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base, indices, tfm): self.base=base; self.indices=list(indices); self.tfm=tfm
    def __len__(self): return len(self.indices)
    def __getitem__(self,i):
        path,y=self.base.samples[self.indices[i]]
        from PIL import Image
        with Image.open(path) as img: return self.tfm(img.convert('RGB')), float(y)

def main():
    ap=argparse.ArgumentParser(description='Train CNN for defective/non-defective surfaces')
    ap.add_argument('--data',default='data/demo')
    ap.add_argument('--epochs',type=int,default=8)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--lr',type=float,default=0.001)
    ap.add_argument('--model',default='models/defect_cnn.pt')
    ap.add_argument('--output',default='outputs')
    ap.add_argument('--device',default='auto',choices=['auto','cuda','cpu'],help='Device to train on (cuda/cpu/auto)')
    args=ap.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    base=ImageFolder(args.data)
    if base.classes!=LABELS:
        # ImageFolder sorts alphabetically, so defective comes before non_defective.
        if base.classes!=['defective','non_defective']:
            raise ValueError('Dataset folders must be defective/ and non_defective/')
        base.samples=[(p,1-y) for p,y in base.samples]
        base.targets=[1-y for y in base.targets]
    targets=np.array([y for _,y in base.samples]); ids=np.arange(len(targets))
    if len(set(targets))!=2 or min(np.bincount(targets))<5: raise ValueError('Need at least 5 images in each class')
    tr,rest=train_test_split(ids,test_size=0.3,stratify=targets,random_state=42)
    va,te=train_test_split(rest,test_size=0.5,stratify=targets[rest],random_state=42)
    loaders={name:DataLoader(IndexedDataset(base,ix,transform(name=='train')),batch_size=args.batch_size,shuffle=name=='train',num_workers=0) for name,ix in [('train',tr),('val',va),('test',te)]}
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    if device == 'cuda':
        print(f"🚀 Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"⚙️ Training on CPU (For T4 GPU acceleration, run on Google Colab / Kaggle with T4 runtime)")
    model=DefectCNN().to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    loss_fn=nn.BCEWithLogitsLoss(); best=float('inf'); history=[]
    modelpath=Path(args.model);modelpath.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(args.epochs):
        scores={}
        for split in ('train','val'):
            model.train(split=='train'); total=0; correct=0; loss_sum=0
            for x,y in loaders[split]:
                x=x.to(device); y=y.to(device)
                with torch.set_grad_enabled(split=='train'):
                    logits=model(x);loss=loss_fn(logits,y)
                    if split=='train': opt.zero_grad();loss.backward();opt.step()
                total+=len(y);loss_sum+=loss.item()*len(y);correct+=((torch.sigmoid(logits)>=0.5)==y.bool()).sum().item()
            scores[split+'_loss']=round(loss_sum/total,5);scores[split+'_accuracy']=round(correct/total,5)
        scores['epoch']=epoch+1;history.append(scores);print(scores)
        if scores['val_loss']<best:
            best=scores['val_loss'];torch.save({'state_dict':model.state_dict(),'labels':LABELS,'image_size':128,'epoch':epoch+1,'demo_data': 'demo' in args.data.lower()},modelpath)
    from src.core import load_model
    model,_=load_model(modelpath,device)
    truth=[];pred=[]
    for x,y in loaders['test']:
        with torch.no_grad(): p=(torch.sigmoid(model(x.to(device)))>=0.5).int().cpu().tolist()
        pred.extend(p);truth.extend(y.int().tolist())
    result={'test_accuracy':accuracy_score(truth,pred),'test_f1':f1_score(truth,pred,zero_division=0),'confusion_matrix':confusion_matrix(truth,pred,labels=[0,1]).tolist(),'classification_report':classification_report(truth,pred,target_names=LABELS,zero_division=0,output_dict=True),'history':history,'test_samples':len(truth),'dataset':args.data,'warning':'Synthetic demo metrics do not indicate real industrial performance' if 'demo' in args.data.lower() else 'Check dataset provenance and leakage before interpreting metrics'}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    print('Saved model:',modelpath,'| metrics:',out/'metrics.json')
if __name__=='__main__':main()
