"""SETA-FL pilot scaffold for CIFAR-10.

This is an experimental scaffold, not a validated statistical guarantee.
The e-process is valid only if the supplied scalar score is conditionally
mean-zero and conditionally 1-sub-Gaussian under the benign-client null.
"""
import argparse, copy, json, math, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*8*8,128), nn.ReLU(), nn.Linear(128,10))
    def forward(self,x): return self.net(x)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def dirichlet_partition(targets, n_clients, alpha, seed):
    rng=np.random.default_rng(seed); targets=np.asarray(targets); out=[[] for _ in range(n_clients)]
    for c in np.unique(targets):
        ids=np.where(targets==c)[0]; rng.shuffle(ids)
        props=rng.dirichlet(np.full(n_clients,alpha))
        cuts=(np.cumsum(props)*len(ids)).astype(int)[:-1]
        chunks=np.split(ids,cuts)
        for k,ch in enumerate(chunks): out[k].extend(ch.tolist())
    for x in out: rng.shuffle(x)
    return out

def flat_update(new, old):
    return torch.cat([(new[k]-old[k]).detach().flatten().cpu() for k in old])

def load_state(model): return {k:v.detach().clone() for k,v in model.state_dict().items()}

def apply_aggregate(global_model, updates, weights):
    base=load_state(global_model); new={k:v.clone() for k,v in base.items()}
    for k in new:
        if new[k].is_floating_point():
            new[k]=base[k]+sum(float(w)*u[k].cpu() for w,u in zip(weights,updates))
    global_model.load_state_dict(new)

def local_train(global_model, loader, device, epochs=1, lr=.03, prox=0.0):
    model=copy.deepcopy(global_model).to(device); ref={k:v.detach().clone() for k,v in model.named_parameters()}
    opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.0)
    model.train()
    for _ in range(epochs):
        for x,y in loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=nn.functional.cross_entropy(model(x),y)
            if prox:
                loss=loss+prox/2*sum((p-ref[n]).pow(2).sum() for n,p in model.named_parameters())
            loss.backward(); opt.step()
    st=model.state_dict(); return {k:(st[k].cpu()-v.cpu()) if st[k].is_floating_point() else torch.zeros_like(v.cpu()) for k,v in global_model.state_dict().items()}

def evaluate(model, loader, device):
    model.eval(); loss=0.; correct=0; total=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); z=model(x); loss+=nn.functional.cross_entropy(z,y,reduction='sum').item(); correct+=(z.argmax(1)==y).sum().item(); total+=len(y)
    return loss/total, correct/total

class MixtureE:
    """Product of predictable mixtures of fixed exponential bets."""
    def __init__(self, lambdas=(.1,.2,.4), alpha=.05):
        self.lambdas=np.asarray(lambdas,dtype=float); self.log_e=0.; self.alpha=alpha; self.t=0
    def update(self,z):
        # Validity requires z_t | F_{t-1} to be 1-sub-Gaussian, mean zero.
        inc=np.concatenate([self.lambdas*z-.5*self.lambdas**2,
                            -self.lambdas*z-.5*self.lambdas**2])
        m=float(inc.max()); log_factor=m+math.log(float(np.exp(inc-m).mean()))
        self.log_e+=log_factor; self.t+=1
        return self.log_e, self.log_e>=math.log(1/self.alpha)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='./data'); ap.add_argument('--out',default='./results'); ap.add_argument('--seed',type=int,default=1); ap.add_argument('--alpha',type=float,default=.5); ap.add_argument('--clients',type=int,default=10); ap.add_argument('--rounds',type=int,default=30); ap.add_argument('--attack',choices=['none','signflip','labelflip'],default='signflip'); ap.add_argument('--drift-round',type=int,default=16); ap.add_argument('--device',default='cpu'); args=ap.parse_args()
    seed_all(args.seed); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); device=torch.device(args.device)
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.247,.243,.261))])
    train=datasets.CIFAR10(args.root,train=True,download=True,transform=tf); test=datasets.CIFAR10(args.root,train=False,download=True,transform=tf)
    parts=dirichlet_partition(train.targets,args.clients,args.alpha,args.seed)
    drift_parts=dirichlet_partition(train.targets,args.clients,args.alpha,args.seed+10000)
    test_loader=DataLoader(test,batch_size=256,shuffle=False); global_model=SmallCNN().to(device)
    opt_e=[MixtureE() for _ in range(args.clients)]; hist=[[] for _ in range(args.clients)]; rows=[]
    for rnd in range(1,args.rounds+1):
        old=load_state(global_model); updates=[]; sizes=[]; scores=[]; alarms=[]
        for i,ids0 in enumerate(parts):
            # Simulate client-level distribution drift by changing the Dirichlet partition
            # after drift_round. This is a controlled synthetic drift, not a natural drift benchmark.
            ids = drift_parts[i] if rnd >= args.drift_round else ids0
            ds=Subset(train,ids)
            loader=DataLoader(ds,batch_size=64,shuffle=True)
            upd=local_train(global_model,loader,device)
            # Apply attack BEFORE computing the detector score.
            if args.attack=='signflip' and i<max(1,args.clients//5) and rnd>=args.drift_round:
                upd={k:-v for k,v in upd.items()}
            elif args.attack=='labelflip' and i<max(1,args.clients//5) and rnd>=args.drift_round:
                raise NotImplementedError('Implement label remapping in a client-specific Dataset wrapper before using labelflip.')
            vec=flat_update({k:old[k]+upd[k] for k in old},old).numpy()
            # Predictable baseline: previous accepted-score median; cold start uses zero.
            baseline=float(np.median(hist[i][-5:])) if hist[i] else 0.
            raw=float(np.linalg.norm(vec)/(math.sqrt(len(vec))+1e-12))
            # Scalar diagnostic only; not a calibrated null residual.
            z=float(np.clip((raw-baseline)/(np.std(hist[i][-5:])+0.05 if len(hist[i])>=3 else 1.0),-1,1))
            loge,alarm=opt_e[i].update(z)
            updates.append(upd); sizes.append(len(ids)); scores.append(loge); alarms.append(int(alarm))
            # This prototype does not update a trusted baseline automatically: that requires a valid acceptance rule.
            if not alarm: hist[i].append(raw)
        # Equal client weights for transparent baseline; trust weights are deliberately not enabled yet.
        weights=np.asarray(sizes,dtype=float); weights/=weights.sum()
        apply_aggregate(global_model,updates,weights)
        loss,acc=evaluate(global_model,test_loader,device)
        rows.append({'round':rnd,'test_loss':loss,'test_accuracy':acc,'alarm_clients':sum(alarms),'mean_log_e':float(np.mean(scores)),'attack':args.attack,'alpha_dirichlet':args.alpha,'seed':args.seed})
        print(f"round={rnd:03d} loss={loss:.4f} acc={acc:.4f} alarms={sum(alarms)}")
    pd.DataFrame(rows).to_csv(out/f'rounds_seed{args.seed}.csv',index=False)
    (out/f'config_seed{args.seed}.json').write_text(json.dumps(vars(args),indent=2))
    print('WARNING: this scaffold tests plumbing only; z is not a calibrated conditional residual and no FPR guarantee is claimed.')
if __name__=='__main__': main()
