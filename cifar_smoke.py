"""Integrity-checked CIFAR-10 import/download and at most 32-image smoke training."""
import argparse
import json
from pathlib import Path
import tarfile

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity

from run_cifar_sequential import SmallCNN, seed_all, local_train, apply_aggregate, evaluate


def import_official_archive(archive,root):
    archive=Path(archive)
    if not check_integrity(str(archive),datasets.CIFAR10.tgz_md5):
        raise ValueError('archive does not match torchvision official CIFAR-10 MD5; not extracted')
    Path(root).mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as handle:
        handle.extractall(root,filter='data')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='./data')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--import-archive')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--out',default='reports/stage3/cifar-smoke.json')
    args=parser.parse_args()
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(dataset='CIFAR-10',official_url=datasets.CIFAR10.url,
                official_archive_md5=datasets.CIFAR10.tgz_md5,root=str(Path(args.root).resolve()),
                download_requested=args.download,full_training_run=False)
    try:
        if args.import_archive:
            import_official_archive(args.import_archive,args.root)
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
            (.4914,.4822,.4465),(.247,.243,.261))])
        train=datasets.CIFAR10(args.root,train=True,download=args.download,transform=transform)
        test=datasets.CIFAR10(args.root,train=False,download=args.download,transform=transform)
        image,label=train[0]
        assert len(train)==50000 and len(test)==10000
        assert image.shape==(3,32,32) and 0<=label<10
        result.update(status='real_data_verified',train_size=len(train),test_size=len(test),image_shape=list(image.shape))
        if not args.check_only:
            seed_all(20261103); torch.set_num_threads(2)
            model=SmallCNN()
            updates=[local_train(model,DataLoader(Subset(train,list(range(start,start+16))),batch_size=8), 'cpu')
                     for start in (0,16)]
            weights=apply_aggregate(model,updates,[16,16])
            loss,accuracy=evaluate(model,DataLoader(Subset(test,list(range(32))),batch_size=8),'cpu')
            result.update(smoke_passed=True,train_images=32,test_images=32,
                          clients=2,sgd_steps=[2,2],global_aggregations=1,weights=weights.tolist(),
                          tiny_subset_loss=loss,tiny_subset_accuracy=accuracy,
                          note='Functionality only; not a benchmark or detector evaluation.')
    except Exception as error:
        result.update(status='blocked',error_type=type(error).__name__,error=str(error),smoke_passed=False)
        out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        raise SystemExit(2)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
