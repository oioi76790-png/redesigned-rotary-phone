"""Check recorded counts, hashes, summaries and optional exact-cell replay."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from stress_validation import run_cell, summarize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--replay-cell',action='store_true')
    args=parser.parse_args()
    base=Path('reports/stage3')
    directories=[(base,Path('configs/stage3.json')),
                 (base/'misspecification',Path('configs/stage3_misspecification.json')),
                 (base/'boundary',Path('configs/stage3_boundary.json'))]
    records=0; cells=0; all_summaries=[]
    for folder,config_path in directories:
        saved=json.loads((folder/'results.json').read_text())
        assert saved['config']==json.loads(config_path.read_text())
        for source,digest in saved['source_sha256'].items():
            path=(folder/'source_snapshot'/source) if source.endswith('.py') else Path(source)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,(folder,source)
        for raw in saved['raw_files']:
            path=folder/raw['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==raw['sha256'],path
            data=pd.read_csv(path)
            assert len(data)==raw['rows']==210000
            assert not data.duplicated(['trial','method','alpha']).any()
            assert data.trial.min()==0 and data.trial.max()==9999
            assert (data.groupby(['method','alpha']).size()==10000).all()
            detector=data.has_detector
            assert data.loc[~detector,'first_alarm'].isna().all()
            assert data.loc[detector,'first_alarm'].isin([-1]+list(range(1,241))).all()
            scenario=data.scenario.iloc[0]; strength=float(data.strength.iloc[0])
            attack=not scenario.startswith('normal_')
            onset=161 if scenario=='reputation_attack' else 41
            rebuilt=summarize(data,attack,onset,240)
            expected=[r for r in saved['summary'] if r['scenario']==scenario and r['strength']==strength]
            assert len(expected)==len(rebuilt)==21
            for actual,previous in zip(rebuilt,expected):
                assert actual['method']==previous['method'] and actual['alpha']==previous['alpha']
                for field in ('false_alarm_rate','detection_rate','mean_delay','restricted_mean_delay',
                              'prediction_mse','aggregation_mse','counterfactual_aggregation_mse',
                              'benign_envelope_violation_rate'):
                    if previous[field] is None:
                        assert actual[field] is None,(path,field)
                    else:
                        assert np.isclose(actual[field],previous[field],rtol=1e-9,atol=1e-12),(path,field)
                for field in ('false_alarm_ci95','detection_ci95'):
                    assert actual[field]==previous[field],(path,field)
            full=data[data.method=='full_seta_candidate'].sort_values(['alpha','trial'])
            fixed=data[data.method=='conditional_seq_fixed'].sort_values(['alpha','trial'])
            np.testing.assert_array_equal(full.first_alarm,fixed.first_alarm)
            all_summaries.extend(expected)
            records+=len(data);cells+=1
        print(f'Verified {folder}: raw hashes, source snapshots and summaries.',flush=True)
    assert cells==19 and records==3990000
    matched=pd.read_csv(base/'matched_prediction.csv.gz')
    assert len(matched)==40000
    iid=matched.scenario.isin(['normal_iid','normal_non_iid'])
    np.testing.assert_array_equal(matched.loc[iid,'history_mse'],matched.loc[iid,'conditional_mse'])
    matched_summary=json.loads((base/'matched_prediction.json').read_text())['summary']
    for row in matched_summary:
        subset=matched[matched.scenario==row['scenario']]
        assert len(subset)==10000
        assert np.isclose(subset.paired_improvement.mean(),row['paired_improvement'])
    log=(base/'unit-tests.log').read_text()
    assert re.search(r'Ran 34 tests',log) and 'OK (skipped=1)' in log
    assert len(list((base/'figures').glob('*.png')))==5
    assert len(list((base/'figures').glob('*.svg')))==5
    replay=False
    if args.replay_cell:
        config=json.loads(Path('configs/stage3_misspecification.json').read_text())
        actual,_,_=run_cell(config,'normal_unobserved_dynamic_drift',0.)
        saved=pd.read_csv(base/'misspecification/normal_unobserved_dynamic_drift_s0.csv.gz')
        np.testing.assert_allclose(actual.first_alarm,saved.first_alarm,equal_nan=True)
        np.testing.assert_allclose(actual.prediction_mse,saved.prediction_mse,rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(actual.aggregation_mse,saved.aggregation_mse,rtol=1e-10,atol=1e-12)
        replay=True
        print('Replayed all 10000 trajectories of an unobserved-drift cell with current code.',flush=True)
    source_files=['run_cifar_sequential.py','seta_statistics.py','calibration.py','stress_validation.py',
                  'matched_ablation.py','cifar_smoke.py','plot_stage3.py','build_stage3_report.py','audit_stage3.py',
                  'tests/test_calibration.py','tests/test_training.py','tests/test_statistics.py']
    source_files+=sorted(str(path) for path in Path('configs').glob('stage3*.json'))
    result=dict(status='passed',raw_rows=records,cells=cells,trials_per_cell=10000,
                paired_prediction_rows=len(matched),summary_rows=len(all_summaries),
                unit_tests=dict(total=34,passed=33,skipped=1,failed=0),full_cell_replayed=replay,
                current_source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_files},
                real_cifar=json.loads((base/'cifar-smoke.json').read_text()))
    (base/'validation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('current_source_sha256','real_cifar')},indent=2))


if __name__=='__main__':
    main()
