"""Frozen-protocol bounded-feature stress tests; no neural training or label tuning.

Each cell has 10,000 independent trajectories, paired across alpha and methods.
External clean references are independent of targets and available one round later.
"""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import time

import numpy as np
import pandas as pd

from calibration import BatchEProcess, fixed_history_bound, rolling_history_bound, conservative_bound
from simulate_statistics import wilson


METHODS=('history_mean','rolling_mean','conditional','conditional_seq_fixed',
         'conditional_seq_rolling','conditional_seq_conservative','full_seta_candidate')


def summarize(frame,attack,onset,horizon):
    result=[]
    for (method,alpha),group in frame.groupby(['method','alpha'],sort=False):
        detector=bool(group.has_detector.iloc[0])
        first=group.first_alarm.to_numpy()
        early=(first>=1)&(first<onset) if attack else first>=1
        at_risk=~early if attack else np.ones(len(group),dtype=bool)
        detected=(first>=onset)&at_risk if attack else np.zeros(len(group),dtype=bool)
        delays=first[detected]-onset+1
        risk_count=int(at_risk.sum())
        false_count=int(early.sum())
        detected_count=int(detected.sum())
        result.append(dict(method=method,alpha=float(alpha),trials=len(group),has_detector=detector,
                           false_alarm_window=f'1..{onset-1}' if attack else f'1..{horizon}',
                           false_alarm_rate=false_count/len(group) if detector else None,
                           false_alarm_ci95=wilson(false_count,len(group)) if detector else None,
                           false_alarms=false_count if detector else None,
                           detection_rate=detected_count/risk_count if detector and attack and risk_count else None,
                           detection_ci95=wilson(detected_count,risk_count) if detector and attack else None,
                           at_risk=risk_count if detector and attack else None,
                           detected=detected_count if detector and attack else None,
                           censored=risk_count-detected_count if detector and attack else None,
                           mean_delay=float(delays.mean()) if len(delays) else None,
                           delay_mcse=float(delays.std(ddof=1)/math.sqrt(len(delays))) if len(delays)>1 else None,
                           restricted_mean_delay=float(np.where(detected[at_risk],first[at_risk]-onset+1,horizon-onset+1).mean()) if detector and attack and risk_count else None,
                           prediction_mse=float(group.prediction_mse.mean()),
                           aggregation_mse=float(group.aggregation_mse.mean()),
                           counterfactual_aggregation_mse=float(group.counterfactual_aggregation_mse.mean()),
                           benign_envelope_violation_rate=float(group.benign_envelope_violation.mean()) if detector else None))
    return result


def run_cell(config,scenario,strength):
    n=config['trials_per_cell']; horizon=config['horizon']
    scenario_id=config['scenarios'].index(scenario)
    # Same target/reference trajectories across attack strengths: paired comparison.
    rng=np.random.default_rng(np.random.SeedSequence([config['seed'],scenario_id]))
    alphas=np.array(config['alphas']); delta=alphas*config['calibration_budget_fraction']
    ealpha=alphas-delta
    attack=not scenario.startswith('normal_')
    onset=config['reputation_onset'] if scenario=='reputation_attack' else config['onset']
    intercept=np.full(n,config['base_intercept'])
    boundary=scenario=='normal_boundary_rademacher'
    if scenario=='normal_non_iid':
        intercept+=rng.uniform(-.2,.1,n)
    slope=config['context_slope']; noise=config['noise_radius']
    if boundary:
        intercept[:]=0.
        slope=0.
    nanchor=config['anchor_samples_per_context']
    c0=-1. if scenario in ('normal_dynamic_drift','normal_gradual_drift') else 0.
    def noise_mean(count):
        if boundary:
            return rng.choice([-1.,1.],(n,count)).mean(axis=1)
        return rng.uniform(-noise,noise,(n,count)).mean(axis=1)
    low=intercept-slope+noise_mean(nanchor)
    high=intercept+slope+noise_mean(nanchor)
    def predict(context):
        return (1-np.asarray(context))*low/2+(1+np.asarray(context))*high/2
    fixed_mean=intercept+slope*c0+noise_mean(config['initial_mean_samples'])
    blocks=config['rolling_window']//config['external_reference_batch']
    contexts=np.full((n,blocks),c0)
    refs=np.stack([intercept+slope*c0+noise_mean(config['external_reference_batch']) for _ in range(blocks)],axis=1)
    eps_fixed=np.array([fixed_history_bound(nanchor,d) for d in delta])
    eps_anchor_rolling=np.array([fixed_history_bound(nanchor,d/2) for d in delta])
    eps_roll=np.array([float(rolling_history_bound(config['rolling_window'],d/2,horizon,e,0.))
                       for d,e in zip(delta,eps_anchor_rolling)])
    monitor=BatchEProcess((n,len(alphas),3),config['lambdas'])
    first=np.full((n,len(alphas),3),-1,dtype=np.int32)
    violations=np.zeros_like(first,dtype=bool)
    prediction_error=np.zeros((n,4))
    aggregation_error=np.zeros(n)
    counterfactual_error=np.zeros(n)
    filtered_error=np.zeros((n,len(alphas)))
    filtered_counterfactual=np.zeros_like(filtered_error)
    mass=config['suspect_sample_mass']
    for t in range(1,horizon+1):
        if scenario=='normal_dynamic_drift':
            context= -1. if t<config['onset'] else (1. if (t-config['onset'])//40%2==0 else -1.)
        elif scenario=='normal_gradual_drift':
            context=-1.+2*(t-1)/(horizon-1)
        else:
            context=0.
        mean=intercept+slope*context
        if scenario=='normal_unobserved_dynamic_drift' and t>=config['onset']:
            mean=mean-config['unobserved_shift_max']
        elif scenario=='normal_unobserved_gradual_drift':
            mean=mean-config['unobserved_shift_max']*(t-1)/(horizon-1)
        # Everything below this line until target generation is predictable.
        conditional=predict(context)
        rolling=refs.mean(axis=1)
        rolling_conditional=conditional+rolling-predict(contexts.mean(axis=1))
        rolling_conditional=np.clip(rolling_conditional,-1.,1.)
        distance=np.abs(contexts-context).mean(axis=1)
        centers=np.stack((conditional,rolling_conditional,conditional),axis=-1)
        bounds=np.empty((n,len(alphas),3))
        bounds[:,:,0]=eps_fixed
        bounds[:,:,1]=np.minimum(2.,eps_roll[None,:]+distance[:,None]*eps_anchor_rolling[None,:])
        bounds[:,:,2]=conservative_bound(conditional)[:,None]
        prediction_error+=(np.stack((fixed_mean,rolling,conditional,rolling_conditional),axis=-1)-mean[:,None])**2
        violations|=abs(mean[:,None,None]-centers[:,None,:])>bounds+1e-12
        clean=rng.choice([-1.,1.],n) if boundary else mean+rng.uniform(-noise,noise,n)
        observed=clean.copy()
        active=attack and t>=onset and (scenario!='intermittent_poisoning' or (t-onset)%4<2)
        if active:
            observed=clean-strength*config['weak_attack_max_shift'] if scenario=='weak_stealth_attack' else (1-2*strength)*clean
        if np.any(abs(observed)>1):
            raise ValueError('DGP left the declared bounded interval; do not silently clip')
        log_e=monitor.update(observed[:,None,None],centers[:,None,:],bounds)
        crossing=log_e>=np.log(1/ealpha)[None,:,None]
        first=np.where((first<0)&crossing,t,first)
        # Independent benign peer updates. Latent means/counterfactual clean targets
        # are used only in evaluation, never in predictions, bounds or alarms.
        others=(rng.choice([-1.,1.],(n,config['clean_other_clients'])).mean(axis=1) if boundary else
                mean+rng.uniform(-noise,noise,(n,config['clean_other_clients'])).mean(axis=1))
        fedavg=mass*observed+(1-mass)*others
        clean_aggregate=mass*clean+(1-mass)*others
        gated=np.where(first[:,:,0]>=1,others[:,None],fedavg[:,None])
        aggregation_error+=(fedavg-mean)**2
        counterfactual_error+=(fedavg-clean_aggregate)**2
        filtered_error+=(gated-mean[:,None])**2
        filtered_counterfactual+=(gated-clean_aggregate[:,None])**2
        # External clean validation is independent of target and attack identity.
        # Created after this round, admitted only for next round's prediction.
        slot=(t-1)%blocks
        contexts[:,slot]=context
        refs[:,slot]=mean+noise_mean(config['external_reference_batch'])
    frames=[]
    for ai,alpha in enumerate(alphas):
        for mi,method in enumerate(METHODS):
            detector=mi>=3
            strategy=mi-3 if 3<=mi<=5 else 0
            pi=mi if mi<3 else (3 if strategy==1 else 2)
            gate=method=='full_seta_candidate'
            frames.append(pd.DataFrame(dict(
                scenario=scenario,strength=strength,trial=np.arange(n),alpha=alpha,
                method=method,has_detector=detector,
                first_alarm=first[:,ai,strategy] if detector else np.full(n,np.nan),
                prediction_mse=prediction_error[:,pi]/horizon,
                aggregation_mse=(filtered_error[:,ai] if gate else aggregation_error)/horizon,
                counterfactual_aggregation_mse=(filtered_counterfactual[:,ai] if gate else counterfactual_error)/horizon,
                benign_envelope_violation=violations[:,ai,strategy] if detector else False,
                final_log_e=log_e[:,ai,strategy] if detector else np.full(n,np.nan))))
    frame=pd.concat(frames,ignore_index=True)
    paired=[]
    for idx,name in enumerate(('history_mean','rolling_mean')):
        improvement=(prediction_error[:,idx]-prediction_error[:,2])/horizon
        se=float(improvement.std(ddof=1)/math.sqrt(n))
        paired.append(dict(scenario=scenario,strength=strength,baseline=name,trials=n,
                           mse_improvement=float(improvement.mean()),
                           paired_ci95=[float(improvement.mean()-1.96*se),float(improvement.mean()+1.96*se)]))
    return frame,summarize(frame,attack,onset,horizon),paired


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/stage3.json')
    parser.add_argument('--out',default='reports/stage3')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    if config['trials_per_cell']<10000 or config['rolling_window']%config['external_reference_batch']:
        parser.error('need >=10000 trials and whole rolling reference blocks')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    summaries=[]; paired=[]; raw_files=[]
    started=time.monotonic()
    for scenario in config['scenarios']:
        strengths=[0.] if scenario.startswith('normal_') else config['attack_strengths']
        for strength in strengths:
            before=time.monotonic()
            frame,summary,comparisons=run_cell(config,scenario,strength)
            name=f'{scenario}_s{strength:g}.csv.gz'
            frame.to_csv(out/name,index=False,compression={'method':'gzip','mtime':0},float_format='%.12g')
            summaries.extend(dict(scenario=scenario,strength=strength,**row) for row in summary)
            paired.extend(comparisons)
            raw_files.append(dict(file=name,rows=len(frame),sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()))
            print(json.dumps(dict(scenario=scenario,strength=strength,trajectories=config['trials_per_cell'],rows=len(frame),seconds=round(time.monotonic()-before,2))),flush=True)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_prediction.csv',index=False)
    sources=['calibration.py','seta_statistics.py','stress_validation.py','configs/stage3.json']
    result=dict(config=config,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                total_seconds=time.monotonic()-started,raw_files=raw_files,
                source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
                summary=summaries,paired_prediction=paired,
                scope='bounded scalar features, independent clean reference service; candidate filtering, no real FL claims')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('All frozen-protocol cells completed.',flush=True)


if __name__=='__main__':
    main()
