"""Gated 36-run real CIFAR pilot; never substitutes synthetic data."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from client_control import ErrorBudget
from monitoring_audit import replay_client_logs


def plan_runs(config):
    runs=[]
    for seed,partition,attack,policy in itertools.product(config['seeds'],config['partitions'],config['attacks'],config['uncalibrated_policies']):
        runs.append(dict(run_id=f'seed{seed}_{partition}_{attack}_{policy}',seed=seed,partition=partition,
                         attack=attack,policy=policy,status='NOT_RUN'))
    if len(runs)!=config['planned_runs'] or len({r['run_id'] for r in runs})!=len(runs) or not 10<=config['rounds']<=20:
        raise ValueError('pilot registry or round count differs from declared protocol')
    if config['independent_monitored_features']!=1 or config['feature_dimension']!=16:
        raise ValueError('current pilot implements exactly one 16-dimensional projected scalar test')
    if config['attack_start']<=config['warmup_rounds'] or config['warmup_rounds']<config['minimum_history']:
        raise ValueError('declared clean warmup must end before attacks and cover calibration minimum')
    return runs


def run_command(config,run,root,out):
    command=[sys.executable,'run_cifar_sequential.py','--root',str(root),'--out',str(out),
             '--offline','--clients',str(config['clients']),'--rounds',str(config['rounds']),
             '--seed',str(run['seed']),'--partition',run['partition'],'--alpha',str(config['dirichlet_alpha']),
             '--attack',run['attack'],'--attack-start',str(config['attack_start']),
             '--malicious-fraction',str(config['malicious_fraction']),'--drift-round',str(config['rounds']+1),
             '--uncalibrated-policy',run['policy'],'--warmup-rounds',str(config['warmup_rounds']),
             '--minimum-history',str(config['minimum_history']),'--bound-mode',config['bound_mode'],
             '--mean-error-bound',str(config['assumed_mean_error_bound']),'--feature-scale',str(config['feature_scale']),
             '--family-alpha',str(config['alpha_global_experiment_family']),'--family-runs',str(config['planned_runs']),
             '--calibration-fraction',str(config['calibration_fraction']),
             '--uncalibrated-weight-cap',str(config['uncalibrated_weight_cap']),
             '--uncalibrated-group-cap',str(config['uncalibrated_group_cap']),
             '--train-limit',str(config['train_limit']),'--test-limit',str(config['test_limit']),
             '--device',config['device'],'--threads',str(config['threads'])]
    if run['policy']=='clean-warmup':
        command.append('--declare-clean-warmup')
    return command


def summarize_run(config,run,directory):
    seed=run['seed']
    rounds=pd.read_csv(directory/f'rounds_seed{seed}.csv')
    clients=pd.read_csv(directory/f'clients_seed{seed}.csv')
    evidence_audit=replay_client_logs(directory/f'clients_seed{seed}.csv')
    if len(rounds)!=config['rounds'] or len(clients)!=config['rounds']*config['clients']:
        raise ValueError('incomplete round/client logs')
    if not np.isfinite(rounds[['test_loss','test_accuracy']].to_numpy()).all():
        raise ValueError('nonfinite training/evaluation metrics')
    mass=clients.groupby('round').weight.sum()+rounds.set_index('round').server_weight
    np.testing.assert_allclose(mass,1.,atol=1e-10)
    monitoring=clients[clients['round']>config['warmup_rounds']]
    monitored_ids=set(monitoring[monitoring.detector_enabled&monitoring.selected].client)
    malicious=set(range(int(np.ceil(config['clients']*config['malicious_fraction'])))) if run['attack']=='signflip' else set()
    benign=set(range(config['clients']))-malicious
    alarm_ids=set(monitoring[monitoring.state_after=='ALARMED'].client)
    benign_monitored=benign&monitored_ids
    malicious_monitored=malicious&monitored_ids
    def ratio(numerator,denominator):
        return numerator/denominator if denominator else None
    return dict(run_id=run['run_id'],seed=seed,partition=run['partition'],attack=run['attack'],policy=run['policy'],
                final_accuracy=float(rounds.test_accuracy.iloc[-1]),final_loss=float(rounds.test_loss.iloc[-1]),
                metrics_scope='tiny real subsets; pooled accuracy is descriptive, not a calibrated research claim',
                benign_total=len(benign),benign_monitored=len(benign_monitored),
                benign_monitoring_coverage=ratio(len(benign_monitored),len(benign)),
                benign_false_alarm_rate_monitored=ratio(len(alarm_ids&benign_monitored),len(benign_monitored)),
                benign_flagged_fraction_all=ratio(len(alarm_ids&benign),len(benign)),
                malicious_total=len(malicious),malicious_monitored=len(malicious_monitored),
                malicious_monitoring_coverage=ratio(len(malicious_monitored),len(malicious)),
                malicious_detection_rate_monitored=ratio(len(alarm_ids&malicious_monitored),len(malicious_monitored)),
                malicious_flagged_fraction_all=ratio(len(alarm_ids&malicious),len(malicious)),
                uncalibrated_clients_final=int(rounds.uncalibrated.iloc[-1]),
                all_metrics_finite=True,weights_conserve_server_mass=True,
                evidence_audit=evidence_audit,
                theory_status='Real conditional mean envelope assumed, NOT certified')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/stage4_pilot.json')
    parser.add_argument('--root',default='./data')
    parser.add_argument('--out',default='reports/stage4')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());runs=plan_runs(config)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    existing=out/'pilot_status.json'
    if existing.exists() and json.loads(existing.read_text()).get('runs_started',0)>0:
        raise RuntimeError('This experiment family already started. Unbudgeted restart is prohibited; stateful resume is not implemented.')
    registry=out/'run_registry.csv'
    pd.DataFrame(runs).to_csv(registry,index=False)
    (out/'frozen_plan.json').write_text(json.dumps(config,indent=2)+'\n')
    budget=ErrorBudget(config['alpha_global_experiment_family'],config['clients'],
                       features=config['independent_monitored_features'],runs=len(runs),
                       calibration_fraction=config['calibration_fraction'])
    (out/'family_budget.json').write_text(json.dumps(budget.as_dict(),indent=2)+'\n')
    status=dict(status='PREPARED',runs_planned=len(runs),runs_started=0,runs_completed=0,
                config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                real_data_gate_passed=False,metrics_available=False)
    def save():
        (out/'pilot_status.json').write_text(json.dumps(status,indent=2)+'\n')
        pd.DataFrame(runs).to_csv(registry,index=False)
    if args.prepare_only:
        save();print(json.dumps(status,indent=2));return
    gate=out/'cifar-smoke.json'
    command=[sys.executable,'cifar_smoke.py','--root',args.root,'--out',str(gate),'--device',config['device']]
    if args.download:
        command.append('--download')
    with (out/'cifar-smoke.log').open('w') as log:
        completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
    result=json.loads(gate.read_text()) if gate.exists() else {}
    if completed.returncode or not (result.get('smoke_passed') and result.get('batch_md5_verified')
                                   and result.get('sgd_steps')==[2,2] and result.get('global_aggregations')==1):
        status.update(status='BLOCKED_AT_REAL_DATA_GATE',reason=result.get('error','real-data gate did not pass'),
                      gate_exit_code=completed.returncode)
        save();print(json.dumps(status,indent=2));raise SystemExit(2)
    status.update(status='RUNNING',real_data_gate_passed=True)
    summaries=[]
    for run in runs:
        directory=out/'pilot_runs'/run['run_id'];directory.mkdir(parents=True,exist_ok=True)
        command=run_command(config,run,args.root,directory)
        (directory/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        run['status']='RUNNING';status['runs_started']+=1;save()
        with (directory/'train.log').open('w') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
        if process.returncode:
            run['status']='FAILED';status.update(status='FAILED_RUN',failed_run=run['run_id'])
            save();raise SystemExit(process.returncode)
        summaries.append(summarize_run(config,run,directory))
        run['status']='COMPLETED';status['runs_completed']+=1
        (out/'pilot_metrics.json').write_text(json.dumps(summaries,indent=2)+'\n')
        pd.DataFrame(summaries).to_csv(out/'pilot_metrics.csv',index=False)
        save();print(f"Completed {run['run_id']}",flush=True)
    status.update(status='COMPLETED',metrics_available=True);save()
    print(json.dumps(status,indent=2))


if __name__=='__main__':
    main()
