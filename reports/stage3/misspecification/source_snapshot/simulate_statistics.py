"""Seeded scalar-feature Monte Carlo, not CIFAR-10/model-performance experiments.

Uses the actual ClientHistory and BoundedMeanEProcess implementations. Every
replication has independent RNG, trusted calibration data and a fresh monitor.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from seta_statistics import BoundedMeanEProcess, ClientHistory


SCENARIOS = (
    'normal_iid', 'normal_non_iid', 'normal_drift_uncovered',
    'normal_drift_covered', 'persistent_signflip', 'intermittent_poisoning',
    'normal_boundary_rademacher',
)


def wilson(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    p = successes/count
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return [max(0.,center-radius), min(1.,center+radius)]


def run_trial(scenario, seed, trial, horizon=240, onset=41):
    scenario_id = SCENARIOS.index(scenario)
    # Paired identical drift data: only the predictable envelope differs.
    if scenario == 'normal_drift_covered':
        scenario_id = SCENARIOS.index('normal_drift_uncovered')
    rng = np.random.default_rng(np.random.SeedSequence([seed, scenario_id, trial]))
    boundary = scenario == 'normal_boundary_rademacher'
    mu = float(rng.choice([-1,1])*rng.uniform(.25,.75)) if scenario == 'normal_non_iid' else .6
    history = ClientHistory(max_history=64)
    calibration = np.tile([-1.,1.],32) if boundary else rng.uniform(mu-.1,mu+.1,64)
    for value in calibration:
        history.add_trusted([value], provenance='simulated independently certified benign pre-monitoring data')
    bias = 0. if boundary else (1. if scenario == 'normal_drift_covered' else .1)
    # In this experiment the independently trusted history is frozen throughout.
    prediction = history.predict(bias)
    monitor = BoundedMeanEProcess(alpha=.05)
    attack = scenario in ('persistent_signflip','intermittent_poisoning')
    first_alarm = None
    max_mean_error = 0.
    for round_id in range(1,horizon+1):
        conditional_mean = 0. if boundary else mu
        if 'drift' in scenario and round_id >= onset:
            conditional_mean = mu-.9*min((round_id-onset+1)/80,1.)
        observation = float(rng.choice([-1.,1.])) if boundary else float(rng.uniform(conditional_mean-.1,conditional_mean+.1))
        poisoned = attack and round_id >= onset and (
            scenario == 'persistent_signflip' or (round_id-onset)%4 < 2)
        if poisoned:
            # True sign inversion of the entire one-dimensional update feature.
            observation = -observation
            conditional_mean = -conditional_mean
        score = prediction.score([observation])
        max_mean_error = max(max_mean_error,abs(conditional_mean*prediction.direction[0]-prediction.mean))
        _, alarm = monitor.update(score,prediction.mean,prediction.mean_error_bound)
        history.observe([observation],round_id=round_id,alarm=alarm)
        if alarm and first_alarm is None:
            first_alarm = round_id
    early = attack and first_alarm is not None and first_alarm < onset
    detected = attack and first_alarm is not None and first_alarm >= onset
    return dict(scenario=scenario,trial=trial,first_alarm=first_alarm,
                false_alarm=bool(early if attack else first_alarm is not None),
                detected=bool(detected),at_risk=bool(attack and not early),
                delay=(first_alarm-onset+1) if detected else None,
                restricted_delay=((first_alarm-onset+1) if detected else horizon-onset+1) if attack and not early else None,
                mean=prediction.mean,mean_error_bound=bias,
                max_actual_mean_error=max_mean_error,
                envelope_holds=bool(max_mean_error <= bias+1e-12),
                final_log_e=monitor.log_e)


def summarize(records, horizon, onset):
    summary = []
    for scenario in SCENARIOS:
        rows = [row for row in records if row['scenario']==scenario]
        count = len(rows)
        if not count:
            continue
        false_alarms = sum(row['false_alarm'] for row in rows)
        at_risk = sum(row['at_risk'] for row in rows)
        detections = sum(row['detected'] for row in rows)
        delays = [row['delay'] for row in rows if row['delay'] is not None]
        restricted = [row['restricted_delay'] for row in rows if row['restricted_delay'] is not None]
        attack = scenario in ('persistent_signflip','intermittent_poisoning')
        summary.append(dict(
            scenario=scenario,trials=count,horizon=horizon,attack_onset=onset if attack else None,
            false_alarm_window=f'1..{onset-1}' if attack else f'1..{horizon}',
            false_alarms=false_alarms,false_alarm_rate=false_alarms/count,
            false_alarm_wilson95=wilson(false_alarms,count),
            at_risk_trials=at_risk if attack else None,
            detections=detections if attack else None,
            detection_rate=detections/at_risk if at_risk else None,
            detection_wilson95=wilson(detections,at_risk),
            mean_delay_detected=float(np.mean(delays)) if delays else None,
            mean_delay_detected_mcse=float(np.std(delays,ddof=1)/math.sqrt(len(delays))) if len(delays)>1 else None,
            restricted_mean_delay=float(np.mean(restricted)) if restricted else None,
            censored_trials=at_risk-detections if attack else None,
            null_envelope_violations=sum(not row['envelope_holds'] for row in rows) if not attack else None))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=20261103)
    parser.add_argument('--horizon',type=int,default=240)
    parser.add_argument('--onset',type=int,default=41)
    parser.add_argument('--out',default='reports')
    args = parser.parse_args()
    if args.trials < 1000 or not 1 < args.onset <= args.horizon or args.seed < 0:
        parser.error('need >=1000 trials, nonnegative seed and 1 < onset <= horizon')
    output = Path(args.out)
    output.mkdir(parents=True,exist_ok=True)
    records = []
    for scenario in SCENARIOS:
        records.extend(run_trial(scenario,args.seed,trial,args.horizon,args.onset) for trial in range(args.trials))
        print(json.dumps(summarize(records,args.horizon,args.onset)[-1]),flush=True)
    with (output/'simulation_trials.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = dict(seed=args.seed,trials_per_scenario=args.trials,horizon=args.horizon,onset=args.onset,
                  alpha=.05,lambdas=[.125,.25,.5,1.,2.],trusted_history_size=64,
                  description='Scalar bounded-feature statistical simulation; no CIFAR-10 or neural-network performance claims.',
                  delay_definition='inclusive rounds since onset, conditional on detection and no pre-onset alarm; nondetections separately censored',
                  restricted_delay_definition='min(delay, horizon-onset+1), at-risk trials, nondetections assigned full window',
                  summary=summarize(records,args.horizon,args.onset))
    (output/'simulation_summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
