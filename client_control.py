"""Predictable contextual monitoring, explicit client states and fixed budgets.

The affine real-FL predictor is empirical. Its mean-error envelope is NOT
certified by the scalar simulation theorem. No non-alarm admits a reference.
"""
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import torch

from seta_statistics import BoundedMeanEProcess, Prediction, bounded_vector


class ClientState(str, Enum):
    UNCALIBRATED='UNCALIBRATED'
    MONITORED='MONITORED'
    ALARMED='ALARMED'


@dataclass(frozen=True)
class ErrorBudget:
    alpha_global: float
    clients: int
    features: int=1
    runs: int=1
    calibration_fraction: float=.1

    def __post_init__(self):
        if not 0<self.alpha_global<1 or not 0<self.calibration_fraction<1:
            raise ValueError('invalid total alpha or calibration fraction')
        if any(not isinstance(n,int) or isinstance(n,bool) or n<1 for n in (self.clients,self.features,self.runs)):
            raise ValueError('clients, independent feature tests and runs must be positive integers')

    @property
    def alpha_e(self):
        return self.alpha_global*(1-self.calibration_fraction)/(self.clients*self.features*self.runs)

    @property
    def delta_calibration(self):
        return self.alpha_global*self.calibration_fraction/(self.clients*self.features*self.runs)

    def as_dict(self):
        return dict(alpha_global=self.alpha_global,clients=self.clients,features=self.features,runs=self.runs,
                    calibration_fraction=self.calibration_fraction,alpha_e_per_test=self.alpha_e,
                    delta_per_test=self.delta_calibration,
                    family_sum=(self.alpha_e+self.delta_calibration)*self.clients*self.features*self.runs,
                    guarantee='Budget arithmetic only; calibration failure bound still requires proof.')


@dataclass(frozen=True)
class RoundContext:
    round_id: int
    value: float
    initial_global_rms: float
    current_global_rms: float
    source: str='pre-update global parameter RMS; no current client update'


def model_rms(model):
    values=[v.detach().double().cpu() for v in model.state_dict().values() if v.is_floating_point()]
    count=sum(v.numel() for v in values)
    if not count:
        raise ValueError('no floating state')
    rms=math.sqrt(sum(float((v*v).sum()) for v in values)/count)
    if not math.isfinite(rms):
        raise ValueError('nonfinite global model')
    return rms


def pre_update_context(model,round_id,initial_global_rms):
    if round_id<1 or not math.isfinite(initial_global_rms) or initial_global_rms<0:
        raise ValueError('invalid context origin')
    current=model_rms(model)
    value=math.tanh(math.log((current+1e-12)/(initial_global_rms+1e-12)))
    return RoundContext(round_id,value,initial_global_rms,current)


class ContextualHistory:
    """Direction-fit/calibration split, explicit scalar context, frozen regression.

    Linear ridge fit is a predictor, not a conditional calibration certificate.
    References must be external or from an explicitly declared attack-free phase.
    """
    def __init__(self,capacity=64):
        from collections import deque
        self.references=deque(maxlen=capacity)
        self.quarantine=deque(maxlen=capacity)
        self.dimension=None

    def vector(self,features):
        vector=bounded_vector(features)
        if self.dimension is not None and len(vector)!=self.dimension:
            raise ValueError('feature dimension changed')
        self.dimension=len(vector)
        return tuple(float(x) for x in vector)

    def admit(self,features,context,*,provenance):
        if not math.isfinite(context) or not -1<=context<=1 or not isinstance(provenance,str) or not provenance.strip():
            raise ValueError('explicit bounded context and reference provenance required')
        self.references.append((self.vector(features),float(context),provenance))

    def observe(self,features,round_id):
        self.quarantine.append((self.vector(features),int(round_id)))

    def predict(self,context,*,minimum=8,bound_mode='assumed',assumed_bound=.2):
        if minimum<4 or bound_mode not in ('assumed','conservative') or not 0<=assumed_bound<=2:
            raise ValueError('invalid prediction configuration')
        if not math.isfinite(context) or not -1<=context<=1:
            raise ValueError('invalid pre-update context')
        if len(self.references)<minimum:
            return None,None
        records=list(self.references)
        split=len(records)//2
        center=np.mean([row[0] for row in records[:split]],axis=0)
        norm=float(np.linalg.norm(center))
        if norm<=1e-12:
            direction=np.zeros(len(center));direction[0]=1.
        else:
            direction=center/norm
        calibration=records[split:]
        contexts=np.array([row[1] for row in calibration])
        design=np.column_stack((np.ones(len(contexts)),contexts))
        targets=np.asarray([row[0] for row in calibration])@direction
        # Fixed ridge coefficient, declared before outcomes; no test-label tuning.
        penalty=np.diag([0.,1e-3])
        theta=np.linalg.solve(design.T@design+penalty,design.T@targets)
        mean=float(np.clip(theta[0]+theta[1]*context,-1.,1.))
        bound=1+abs(mean) if bound_mode=='conservative' else float(assumed_bound)
        details=dict(theta=theta.tolist(),direction_samples=split,calibration_samples=len(calibration),
                     context_rank=int(np.linalg.matrix_rank(design)),context_min=float(contexts.min()),
                     context_max=float(contexts.max()),extrapolation=bool(context<contexts.min() or context>contexts.max()),
                     certificate='deterministic_bounded_envelope' if bound_mode=='conservative' else 'ASSUMED_NOT_CERTIFIED')
        return Prediction(tuple(direction),mean,bound,len(records)),details


@dataclass(frozen=True)
class MonitoringSnapshot:
    context: RoundContext
    prediction: object
    details: object
    state_before: ClientState


class ClientController:
    def __init__(self,budget,*,policy='defer',minimum=8,assumed_bound=.2,bound_mode='assumed',
                 warmup_rounds=8,declare_clean_warmup=False,weight_cap=.02,group_cap=.2):
        if budget.features!=1:
            raise ValueError('controller currently implements one scalar test; sketch dimension is not test count')
        if policy not in ('defer','capped','clean-warmup') or minimum<4:
            raise ValueError('invalid state policy or calibration minimum')
        if policy=='clean-warmup' and (not declare_clean_warmup or warmup_rounds<minimum):
            raise ValueError('clean-warmup requires predeclared attack-free rounds covering minimum references')
        if not 0<weight_cap<=1 or not 0<group_cap<=1 or not 0<=assumed_bound<=2 or bound_mode not in ('assumed','conservative'):
            raise ValueError('invalid cap or bound')
        self.budget=budget;self.policy=policy;self.minimum=minimum
        self.assumed_bound=assumed_bound;self.bound_mode=bound_mode
        self.warmup_rounds=warmup_rounds;self.declare_clean_warmup=declare_clean_warmup
        self.weight_cap=weight_cap;self.group_cap=group_cap
        self.histories=[ContextualHistory() for _ in range(budget.clients)]
        self.monitors=[BoundedMeanEProcess(alpha=budget.alpha_e) for _ in self.histories]
        self.states=[ClientState.UNCALIBRATED for _ in self.histories]
        self.snapshots=[];self.logs={};self.round_id=0;self.observed=set();self.warmup_admitted=set()

    def add_external(self,client,features,context,*,provenance):
        if self.round_id:
            raise ValueError('external startup references must precede the monitoring run')
        self.histories[client].admit(features,context,provenance=provenance)

    def is_warmup(self):
        return self.policy=='clean-warmup' and self.round_id<=self.warmup_rounds

    def prepare_round(self,context):
        if context.round_id!=self.round_id+1:
            raise ValueError('rounds must advance exactly once; no evidence reset')
        self.round_id=context.round_id;self.logs={};self.observed=set();self.warmup_admitted=set()
        snapshots=[]
        for client,history in enumerate(self.histories):
            prediction,details=history.predict(context.value,minimum=self.minimum,bound_mode=self.bound_mode,
                                                assumed_bound=self.assumed_bound)
            if self.states[client]!=ClientState.ALARMED:
                self.states[client]=ClientState.MONITORED if prediction is not None and not self.is_warmup() else ClientState.UNCALIBRATED
            snapshots.append(MonitoringSnapshot(context,prediction,details,self.states[client]))
        self.snapshots=tuple(snapshots)
        return self.snapshots

    def observe(self,client,features):
        if not self.snapshots or client in self.observed:
            raise ValueError('must prepare before observing; only one update per client per round')
        self.observed.add(client)
        snapshot=self.snapshots[client];prediction=snapshot.prediction
        score=None;monitor=self.monitors[client]
        if prediction is not None:
            score=prediction.score(features)
        if snapshot.state_before==ClientState.MONITORED:
            _,alarm=monitor.update(score,prediction.mean,prediction.mean_error_bound)
            if alarm:
                self.states[client]=ClientState.ALARMED
        self.histories[client].observe(features,self.round_id)
        self.logs[client]=dict(score=score,log_e=monitor.log_e if monitor.t else None,
                               prediction_mean=prediction.mean if prediction else None,
                               mean_error_bound=prediction.mean_error_bound if prediction else None)
        return self.states[client]==ClientState.ALARMED

    def admit_declared_warmup(self,client,features):
        if not self.is_warmup() or not self.declare_clean_warmup or client not in self.observed or client in self.warmup_admitted:
            raise ValueError('only explicit predeclared pre-monitoring warmup admission is allowed')
        if self.states[client]==ClientState.ALARMED:
            raise ValueError('alarmed client cannot provide a normal reference')
        self.warmup_admitted.add(client)
        self.histories[client].admit(features,self.snapshots[client].context.value,
                                    provenance=f'DECLARED_ATTACK_FREE_PROTOCOL round={self.round_id}; not independently certified')

    def allocate(self,selected,sizes,alarms):
        if not self.snapshots:
            raise ValueError('prepare client states before allocation')
        if len(selected)!=len(sizes) or len(sizes)!=len(alarms):
            raise ValueError('allocation lengths differ')
        if any(int(size)!=size or size<=0 for size in sizes):
            raise ValueError('positive participant sample counts required')
        if not selected:
            return np.array([]),1.
        sizes=np.asarray(sizes,dtype=float)
        allowed=np.array([not alarm and self.states[c]!=ClientState.ALARMED for c,alarm in zip(selected,alarms)])
        unknown=np.array([self.states[c]==ClientState.UNCALIBRATED for c in selected])&allowed
        monitored=np.array([self.states[c]==ClientState.MONITORED for c in selected])&allowed
        weights=np.zeros(len(selected))
        if self.is_warmup():
            accepted=allowed
            if accepted.any():
                weights[accepted]=sizes[accepted]/sizes[accepted].sum()
        else:
            if self.policy=='capped':
                weights[unknown]=np.minimum(sizes[unknown]/sizes.sum(),self.weight_cap)
                if weights.sum()>self.group_cap:
                    weights*=self.group_cap/weights.sum()
            if monitored.any():
                weights[monitored]=(1-weights.sum())*sizes[monitored]/sizes[monitored].sum()
        return weights,max(0.,1-float(weights.sum()))

    def row(self,client,*,selected,weight):
        snapshot=self.snapshots[client]
        details=snapshot.details or {}
        references=self.histories[client].references
        provenance_status=('DECLARED_PROTOCOL_ONLY' if any(r[2].startswith('DECLARED_ATTACK_FREE_PROTOCOL') for r in references)
                           else 'EXTERNAL_ASSERTION_NOT_AUTHENTICATED' if references else 'NO_REFERENCE')
        return dict(round=self.round_id,client=client,selected=selected,
                    state_before=snapshot.state_before.value,state_after=self.states[client].value,
                    weight=float(weight),policy=self.policy,phase='DECLARED_CLEAN_WARMUP' if self.is_warmup() else 'MONITORING',
                    detector_enabled=snapshot.state_before==ClientState.MONITORED,
                    trusted_count=len(self.histories[client].references),quarantine_count=len(self.histories[client].quarantine),
                    context_value=snapshot.context.value,context_rms=snapshot.context.current_global_rms,
                    context_source=snapshot.context.source,
                    alpha_e=self.budget.alpha_e,delta_calibration=self.budget.delta_calibration,
                    calibration_status=details.get('certificate','UNCALIBRATED'),
                    reference_provenance_status=provenance_status,
                    direction=list(snapshot.prediction.direction) if snapshot.prediction else None,
                    theta0=details.get('theta',[None,None])[0],theta1=details.get('theta',[None,None])[1],
                    direction_samples=details.get('direction_samples',0),calibration_samples=details.get('calibration_samples',0),
                    process_steps=self.monitors[client].t,
                    context_rank=details.get('context_rank'),context_extrapolation=details.get('extrapolation'),
                    **self.logs.get(client,dict(score=None,log_e=None,prediction_mean=None,mean_error_bound=None)))
