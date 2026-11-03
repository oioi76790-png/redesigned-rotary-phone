"""Integrity-checked CIFAR-10 import/download and at most 32-image smoke training."""
import argparse
import json
from pathlib import Path
import tarfile
import hashlib
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity

from run_cifar_sequential import SmallCNN, seed_all, local_train, apply_aggregate, evaluate, load_state


def import_official_archive(archive,root):
    archive=Path(archive)
    if not check_integrity(str(archive),datasets.CIFAR10.tgz_md5):
        raise ValueError('archive does not match torchvision official CIFAR-10 MD5; not extracted')
    Path(root).mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as handle:
        if any((Path(root)/member.name).is_file() for member in handle.getmembers()):
            raise ValueError('import destination has existing files; use a fresh root to preserve them')
        handle.extractall(root,filter='data')


def verify_files(root):
    records=[]
    official=datasets.CIFAR10
    entries=official.train_list+official.test_list+[[official.meta['filename'],official.meta['md5']]]
    for filename,expected in entries:
        path=Path(root)/official.base_folder/filename
        if not check_integrity(str(path),expected):
            raise ValueError(f'CIFAR-10 batch integrity check failed: {filename}')
        records.append(dict(file=filename,expected_md5=expected,verified=True))
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='./data')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--import-archive')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--out',default='reports/stage4/cifar-smoke.json')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(dataset='CIFAR-10',official_url=datasets.CIFAR10.url,
                official_archive_md5=datasets.CIFAR10.tgz_md5,root=str(Path(args.root).resolve()),
                download_requested=args.download,full_training_run=False)
    result.update(protocol='32 real train images / 2 clients / 16 each / batch8 / 2 SGD steps each / 1 aggregation',
                  smoke_passed=False,archive_md5_verified=False,batch_md5_verified=False)
    try:
        if args.import_archive:
            import_official_archive(args.import_archive,args.root)
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
            (.4914,.4822,.4465),(.247,.243,.261))])
        train=datasets.CIFAR10(args.root,train=True,download=args.download,transform=transform)
        test=datasets.CIFAR10(args.root,train=False,download=args.download,transform=transform)
        result['batch_checks']=verify_files(args.root)
        result['batch_md5_verified']=True
        archive=Path(args.root)/'cifar-10-python.tar.gz'
        if args.import_archive:
            archive=Path(args.import_archive)
        if archive.exists():
            result['archive_md5_verified']=check_integrity(str(archive),datasets.CIFAR10.tgz_md5)
            if not result['archive_md5_verified']:
                raise ValueError('present archive does not match official MD5')
        else:
            result['archive_md5_status']='not available; official batch MD5s checked individually'
        image,label=train[0]
        assert len(train)==50000 and len(test)==10000
        assert image.shape==(3,32,32) and 0<=label<10
        assert all(0<=int(y)<10 for y in train.targets+test.targets)
        result.update(status='real_data_verified',train_size=len(train),test_size=len(test),image_shape=list(image.shape))
        if not args.check_only:
            seed_all(20261103); torch.set_num_threads(2)
            device=torch.device(args.device)
            model=SmallCNN().to(device)
            device=next(model.parameters()).device
            before=load_state(model)
            updates=[];client_stats=[]
            for start in (0,16):
                stats={}
                loader=DataLoader(Subset(train,list(range(start,start+16))),batch_size=8)
                batch,labels=next(iter(loader))
                assert batch.shape==(8,3,32,32) and labels.dtype==torch.int64
                update=local_train(model,loader,device,stats=stats)
                assert stats['optimizer_steps']==2
                assert stats['model_device']==stats['image_device']==stats['label_device']
                assert all(torch.isfinite(v).all() and v.device.type=='cpu' for v in update.values())
                assert any(torch.count_nonzero(v) for v in update.values())
                for key,value in model.state_dict().items():
                    torch.testing.assert_close(value,before[key],rtol=0,atol=0)
                updates.append(update);client_stats.append(stats)
            expected={key:before[key]+.5*updates[0][key].to(device)+.5*updates[1][key].to(device) for key in before}
            weights=apply_aggregate(model,updates,[16,16])
            np.testing.assert_allclose(weights,[.5,.5])
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,expected[key],rtol=1e-5,atol=1e-7)
                assert value.device==device
            max_error=max(float((model.state_dict()[key]-expected[key]).abs().max().cpu()) for key in expected)
            loss,accuracy=evaluate(model,DataLoader(Subset(test,list(range(32))),batch_size=8),device)
            assert np.isfinite(loss) and 0<=accuracy<=1
            result.update(smoke_passed=True,train_images=32,test_images=32,
                          clients=2,sgd_steps=[2,2],global_aggregations=1,weights=weights.tolist(),
                          actual_client_stats=client_stats,max_aggregation_error=max_error,device=str(device),
                          model_unchanged_by_local_training=True,model_data_device_checks=True,
                          dataset_fingerprint=hashlib.sha256(json.dumps(result['batch_checks'],sort_keys=True).encode()).hexdigest(),
                          tiny_subset_loss=loss,tiny_subset_accuracy=accuracy,
                          note='Functionality only; not a benchmark or detector evaluation.')
    except Exception as error:
        result.update(status='failed' if result.get('status')=='real_data_verified' else 'blocked',
                      error_type=type(error).__name__,error=str(error),smoke_passed=False)
        out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        raise SystemExit(2)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
