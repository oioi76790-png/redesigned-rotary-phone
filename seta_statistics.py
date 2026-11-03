"""Bounded-score e-processes; estimated centers are not validity certificates."""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np


def logmeanexp(values):
    values = np.asarray(values, dtype=float)
    maximum = float(values.max())
    return maximum + math.log(float(np.exp(values-maximum).mean()))


class BoundedMeanEProcess:
    """Valid IF X in [-1,1] and |E[X|past]-mean| <= bias predictably.

    A mixture of cumulative fixed-lambda bets, not a product of per-step mixtures.
    mean and mean_error_bound must be fixed BEFORE observing the current score.
    One instance covers the entire monitoring run; never reset on detected drift.
    """
    def __init__(self, alpha=.05, lambdas=(.125, .25, .5, 1., 2.)):
        positive = np.asarray(lambdas, dtype=float)
        if not 0 < alpha < 1:
            raise ValueError('alpha must lie in (0,1)')
        if positive.ndim != 1 or not len(positive) or not np.all(np.isfinite(positive)) or np.any(positive <= 0):
            raise ValueError('lambdas must be finite positive values')
        self.alpha = float(alpha)
        self.lambdas = np.concatenate((positive, -positive))
        self.log_components = np.zeros(len(self.lambdas))
        self.log_e, self.t, self.alarmed = 0., 0, False

    def update(self, score, mean, mean_error_bound):
        if not all(math.isfinite(v) for v in (score, mean, mean_error_bound)):
            raise ValueError('score, mean and bound must be finite')
        if not -1 <= score <= 1 or not -1 <= mean <= 1 or not 0 <= mean_error_bound <= 2:
            raise ValueError('score/mean must be in [-1,1]; bound in [0,2]')
        self.log_components += (self.lambdas*(score-mean)
                                - np.abs(self.lambdas)*mean_error_bound
                                - .5*self.lambdas**2)
        self.t += 1
        self.log_e = logmeanexp(self.log_components)
        self.alarmed = self.alarmed or self.log_e >= math.log(1/self.alpha)
        return self.log_e, self.alarmed


def bounded_vector(features):
    vector = np.asarray(features, dtype=float)
    if vector.ndim != 1 or not len(vector) or not np.all(np.isfinite(vector)):
        raise ValueError('features must be a nonempty finite vector')
    if np.linalg.norm(vector) > 1+1e-10:
        raise ValueError('feature norm must be <= 1')
    return vector


@dataclass(frozen=True)
class Prediction:
    direction: tuple
    mean: float
    mean_error_bound: float
    trusted_count: int

    def score(self, features):
        features = bounded_vector(features)
        if len(features) != len(self.direction):
            raise ValueError('feature dimension changed')
        return float(np.clip(np.dot(features, self.direction), -1., 1.))


class ClientHistory:
    """Quarantine observations; admit trusted data only by external validation.

    A non-alarm is NOT a trust decision. Provenance strings are audit metadata,
    not authentication; callers are responsible for validating their source.
    """
    def __init__(self, max_history=64):
        if max_history < 2:
            raise ValueError('max_history must be >= 2')
        self.trusted = deque(maxlen=max_history)
        self.quarantine = deque(maxlen=max_history)
        self.dimension = None

    def _vector(self, features):
        vector = bounded_vector(features)
        if self.dimension is not None and len(vector) != self.dimension:
            raise ValueError('feature dimension changed')
        self.dimension = len(vector)
        return tuple(float(v) for v in vector)

    def add_trusted(self, features, *, provenance):
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError('external validation provenance required')
        self.trusted.append((self._vector(features), provenance))

    def observe(self, features, *, round_id, alarm):
        self.quarantine.append((self._vector(features), int(round_id), bool(alarm)))

    def predict(self, mean_error_bound, min_history=8):
        if min_history < 2 or not math.isfinite(mean_error_bound) or not 0 <= mean_error_bound <= 2:
            raise ValueError('invalid minimum history or error bound')
        if len(self.trusted) < min_history:
            return None
        matrix = np.asarray([record[0] for record in self.trusted])
        center = matrix.mean(axis=0)
        norm = float(np.linalg.norm(center))
        if norm <= 1e-12:
            direction = np.zeros(matrix.shape[1])
            direction[0] = 1.
        else:
            direction = center/norm
        mean = float(np.clip(np.mean(matrix @ direction), -1., 1.))
        return Prediction(tuple(direction), mean, float(mean_error_bound), len(matrix))
