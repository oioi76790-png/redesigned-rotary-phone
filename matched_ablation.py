"""Same-data prediction-only control; no sequential test or parameter fitting."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    specification=json.loads(Path('configs/stage3_matched_ablation.json').read_text())
    config=json.loads(Path(specification['source_config']).read_text())
    records=[]; summary=[]
    for scenario in specification['scenarios']:
        n=specification['trials']; horizon=config['horizon']
        rng=np.random.default_rng(np.random.SeedSequence([specification['seed'],config['scenarios'].index(scenario)]))
        intercept=np.full(n,config['base_intercept'])
        if scenario=='normal_non_iid':
            intercept+=rng.uniform(-.2,.1,n)
        noise=config['noise_radius']; size=config['anchor_samples_per_context']
        low=intercept-config['context_slope']+rng.uniform(-noise,noise,(n,size)).mean(axis=1)
        high=intercept+config['context_slope']+rng.uniform(-noise,noise,(n,size)).mean(axis=1)
        unconditional=(low+high)/2
        history_error=np.zeros(n); conditional_error=np.zeros(n)
        for t in range(1,horizon+1):
            if scenario=='normal_dynamic_drift':
                context=-1. if t<config['onset'] else (1. if (t-config['onset'])//40%2==0 else -1.)
            elif scenario=='normal_gradual_drift':
                context=-1+2*(t-1)/(horizon-1)
            else:
                context=0.
            prediction=(1-context)*low/2+(1+context)*high/2
            true_mean=intercept+config['context_slope']*context
            history_error+=(unconditional-true_mean)**2
            conditional_error+=(prediction-true_mean)**2
        history_error/=horizon; conditional_error/=horizon
        difference=history_error-conditional_error
        standard_error=float(difference.std(ddof=1)/math.sqrt(n))
        records.append(pd.DataFrame(dict(scenario=scenario,trial=np.arange(n),
                                         history_mse=history_error,conditional_mse=conditional_error,
                                         paired_improvement=difference)))
        summary.append(dict(scenario=scenario,trials=n,training_samples_per_method=2*size,
                            history_mse=float(history_error.mean()),conditional_mse=float(conditional_error.mean()),
                            paired_improvement=float(difference.mean()),
                            paired_ci95=[float(difference.mean()-1.96*standard_error),float(difference.mean()+1.96*standard_error)]))
    out=Path('reports/stage3');out.mkdir(parents=True,exist_ok=True)
    pd.concat(records,ignore_index=True).to_csv(out/'matched_prediction.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    (out/'matched_prediction.json').write_text(json.dumps(dict(specification=specification,summary=summary),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
