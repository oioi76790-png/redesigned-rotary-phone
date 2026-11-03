"""Replay logged scalar evidence; checks computation, NOT conditional validity."""
import numpy as np
import pandas as pd
from seta_statistics import BoundedMeanEProcess


def replay_client_logs(path):
    frame=pd.read_csv(path)
    processes=0;observations=0
    for client,rows in frame.groupby('client'):
        rows=rows.sort_values('round')
        alpha=rows.alpha_e.iloc[0]
        if not np.allclose(rows.alpha_e,alpha,rtol=0,atol=1e-16):
            raise ValueError('client error budget changed during the monitoring run')
        process=BoundedMeanEProcess(alpha=alpha)
        for row in rows.itertuples():
            if row.selected and row.detector_enabled:
                actual,_=process.update(row.score,row.prediction_mean,row.mean_error_bound)
                if not np.isclose(actual,row.log_e,rtol=1e-10,atol=1e-10):
                    raise ValueError(f'e-process recurrence differs for client {client}, round {row.round}')
                observations+=1
            if int(row.process_steps)!=process.t:
                raise ValueError('missing, duplicated or reset evidence steps')
            if process.alarmed and row.state_after!='ALARMED':
                raise ValueError('threshold crossing did not latch ALARMED')
            if row.state_after=='UNCALIBRATED' and row.detector_enabled:
                raise ValueError('uncalibrated client was tested as calibrated')
        processes+=int(process.t>0)
    return dict(status='passed',client_processes_with_observations=processes,scalar_bets_replayed=observations,
                limitation='Numerical/log consistency only; does not certify conditional mean bounds.')
