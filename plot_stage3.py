"""Plot saved outcomes only; never feeds outcomes back to test parameters."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    out=Path('reports/stage3/figures');out.mkdir(parents=True,exist_ok=True)
    base=Path('reports/stage3')
    data=[]
    for directory in (base,base/'misspecification',base/'boundary'):
        result=json.loads((directory/'results.json').read_text())
        data.extend(result['summary'])
    data=pd.DataFrame(data)
    plt.rcParams.update({'font.size':9,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('png','svg'):
            fig.savefig(out/f'{name}.{suffix}',bbox_inches='tight')
        plt.close(fig)
    labels={'conditional_seq_fixed':'fixed calibration','conditional_seq_rolling':'rolling calibration',
            'conditional_seq_conservative':'conservative envelope'}
    scenarios=['normal_iid','normal_non_iid','normal_dynamic_drift','normal_gradual_drift',
               'normal_unobserved_dynamic_drift','normal_unobserved_gradual_drift','normal_boundary_rademacher']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for axis,(method,label) in zip(axes,labels.items()):
        for scenario in scenarios:
            rows=data[(data.scenario==scenario)&(data.method==method)].sort_values('alpha')
            cis=np.array(rows.false_alarm_ci95.tolist())
            values=rows.false_alarm_rate.to_numpy()
            axis.errorbar(rows.alpha,values,yerr=np.vstack((values-cis[:,0],cis[:,1]-values)),
                          marker='.',capsize=2,label=scenario.removeprefix('normal_'))
        axis.plot([.01,.10],[.01,.10],'k--',label='nominal alpha')
        axis.set(title=label,xlabel='Nominal alpha',ylim=(-.025,1.04))
    axes[0].set_ylabel('P(any alarm by round 240), 95% Wilson CI')
    axes[-1].legend(fontsize=7,loc='center left',bbox_to_anchor=(1.,.5))
    save(fig,'false_alarm_control')

    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    attacks=['persistent_signflip','intermittent_poisoning','reputation_attack','weak_stealth_attack']
    for axis,scenario in zip(axes.flat,attacks):
        for method in ('conditional_seq_fixed','conditional_seq_rolling'):
            for alpha,style in ((.01,':'),(.05,'-'),(.10,'--')):
                rows=data[(data.scenario==scenario)&(data.method==method)&(data.alpha==alpha)].sort_values('strength')
                axis.plot(rows.strength,rows.detection_rate,style,marker='.',label=f'{labels[method]}, alpha={alpha}')
        axis.set(title=scenario,xlabel='Attack strength',ylabel='Detection rate',ylim=(-.025,1.04))
    axes.flat[-1].legend(fontsize=6)
    save(fig,'detection_strength')

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in ('conditional_seq_fixed','conditional_seq_rolling'):
        rows=data[(data.strength==1)&(data.alpha==.05)&(data.method==method)]
        axes[0].plot(rows.scenario,rows.mean_delay,marker='o',label=labels[method])
        axes[1].plot(rows.scenario,rows.restricted_mean_delay,marker='o',label=labels[method])
    for axis,title in zip(axes,('Delay conditional on detection','Restricted mean delay (misses censored)')):
        axis.set(title=title,ylabel='Rounds since onset (inclusive)')
        axis.tick_params(axis='x',rotation=25)
        axis.legend(fontsize=7)
    save(fig,'detection_delay')

    matched=json.loads((base/'matched_prediction.json').read_text())['summary']
    fig,axis=plt.subplots(figsize=(9,4))
    positions=np.arange(len(matched));width=.36
    axis.bar(positions-width/2,[r['history_mse'] for r in matched],width,label='unconditional mean, same 2048 samples')
    axis.bar(positions+width/2,[r['conditional_mse'] for r in matched],width,label='conditional predictor, same samples')
    axis.set_xticks(positions,[r['scenario'].removeprefix('normal_') for r in matched])
    axis.set(yscale='log',ylabel='Prediction MSE against benign conditional mean',title='Same-data prediction ablation; no sequential detector')
    axis.legend(fontsize=8)
    save(fig,'matched_prediction')

    fig,axis=plt.subplots(figsize=(9,4))
    positions=np.arange(len(attacks))
    for offset,method,label in ((-.18,'conditional_seq_fixed','detector only / FedAvg'),(.18,'full_seta_candidate','candidate alarm filter')):
        rows=data[(data.strength==1)&(data.alpha==.05)&(data.method==method)].set_index('scenario')
        axis.bar(positions+offset,[rows.loc[s,'aggregation_mse'] for s in attacks],.36,label=label)
    axis.set_xticks(positions,attacks,rotation=15)
    axis.set(ylabel='Scalar aggregation MSE',title='Candidate filter: scalar proxy only, not FL model accuracy')
    axis.legend()
    save(fig,'candidate_aggregation')
    print('Saved 5 figures in PNG and SVG.')


if __name__=='__main__':
    main()
