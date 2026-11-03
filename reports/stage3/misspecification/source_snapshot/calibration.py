"""Calibration certificates require external clean samples and structural assumptions.

No empirical standard deviation or quantile is used as a conditional-mean bound.
These helpers support the scalar fixed-direction validation experiment. They do
not certify the adaptive sketch in real federated learning.
"""
from dataclasses import dataclass
import math
import numpy as np


def hoeffding_radius(n, delta, comparisons=1, width=2.):
    """Two-sided mean radius for independent bounded observations, union budget.

    Each observation must have support width <= width. Across comparisons use a
    union bound; independence between comparisons is not required.
    """
    if n < 1 or comparisons < 1 or not 0 < delta < 1 or not math.isfinite(width) or width <= 0:
        raise ValueError('invalid calibration parameters')
    return width*math.sqrt(math.log(2*comparisons/delta)/(2*n))


def fixed_history_bound(n_per_anchor, delta):
    """Two independent mean anchors at contexts -1,+1, affine interpolation.

    At c in [-1,1] the nonnegative interpolation weights sum to one, so a
    simultaneous anchor error <= epsilon implies uniform error <= epsilon.
    Requires affine benign conditional mean, not merely a good fitted curve.
    """
    return min(2., hoeffding_radius(n_per_anchor,delta,comparisons=2))


def rolling_history_bound(n, delta, horizon, anchor_error, context_distance):
    """Independent external clean residual window; valid for a fixed horizon.

    For affine residual means slope error <= anchor_error. Account for historical
    context mismatch by anchor_error * average |current_context-reference_context|.
    delta here budgets only rolling errors; the anchor failure budget is separate.
    """
    distance = np.asarray(context_distance,dtype=float)
    if np.any(~np.isfinite(distance)) or np.any(distance < 0) or np.any(distance > 2):
        raise ValueError('context distance must lie in [0,2]')
    if not math.isfinite(anchor_error) or not 0 <= anchor_error <= 2:
        raise ValueError('invalid anchor error')
    return np.minimum(2.,hoeffding_radius(n,delta,comparisons=horizon)+anchor_error*distance)


def conservative_bound(predicted_mean):
    """Deterministic envelope for ANY conditional mean in [-1,1]; low power."""
    mean=np.asarray(predicted_mean,dtype=float)
    if np.any(~np.isfinite(mean)) or np.any(abs(mean)>1):
        raise ValueError('predicted mean outside [-1,1]')
    return 1.+abs(mean)


@dataclass(frozen=True)
class ExternalReference:
    value: float
    context: float
    observed_round: int
    available_round: int
    provenance: str

    def __post_init__(self):
        if not -1 <= self.value <= 1 or not -1 <= self.context <= 1:
            raise ValueError('value/context must lie in [-1,1]')
        if self.available_round <= self.observed_round or not self.provenance.strip():
            raise ValueError('require delayed external validation provenance')


class ExternalCleanWindow:
    """Current target updates never enter this window, even without alarms.

    Admission is explicit; provenance text itself is NOT authentication. Caller
    must supply genuinely independent validated reference data.
    """
    def __init__(self,capacity=1024):
        from collections import deque
        if capacity < 1:
            raise ValueError('positive capacity required')
        self.records=deque(maxlen=capacity)

    def admit(self,reference,*,current_round):
        if not isinstance(reference,ExternalReference) or reference.available_round>current_round:
            raise ValueError('reference is not externally available yet')
        self.records.append(reference)

    def snapshot(self):
        return tuple(self.records)


class BatchEProcess:
    """Vectorized version of BoundedMeanEProcess for independent replications.

    Leading dimensions are arbitrary (trial, alpha, strategy). Fixed lambdas and
    predictable per-replication centers/bounds; generic X range [-1,1], proxy=1.
    No thresholds or parameters are chosen from attack labels.
    """
    def __init__(self,shape,lambdas=(.125,.25,.5,1.,2.)):
        positive=np.asarray(lambdas,dtype=float)
        if positive.ndim!=1 or not len(positive) or np.any(~np.isfinite(positive)) or np.any(positive<=0):
            raise ValueError('finite positive lambdas required')
        self.lambdas=np.concatenate((positive,-positive))
        self.log_components=np.zeros(tuple(shape)+(len(self.lambdas),))

    def update(self,score,mean,bound):
        score,mean,bound=np.broadcast_arrays(score,mean,bound)
        if (np.any(~np.isfinite(score)) or np.any(~np.isfinite(mean)) or np.any(~np.isfinite(bound))
                or np.any(abs(score)>1) or np.any(abs(mean)>1) or np.any(bound<0) or np.any(bound>2)):
            raise ValueError('invalid score, center or envelope')
        self.log_components+=(self.lambdas*(score-mean)[...,None]
                              -abs(self.lambdas)*bound[...,None]-.5*self.lambdas**2)
        maximum=self.log_components.max(axis=-1)
        return maximum+np.log(np.exp(self.log_components-maximum[...,None]).mean(axis=-1))
