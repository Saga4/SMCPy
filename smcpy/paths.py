import abc
import numpy as np
import warnings

# compatible with Python 2 *and* 3:
ABC = abc.ABCMeta("ABC", (object,), {"__slots__": ()})


class PathBase:
    def __init__(self, proposal):
        self._phi_list = [0]
        self._proposal = proposal

    @property
    def phi(self):
        return self._phi_list[-1]

    @phi.setter
    def phi(self, phi):
        if phi <= self._phi_list[-1]:
            raise ValueError(
                "phi updates must be monotonic; " f"tried {self.phi} -> {phi}"
            )
        self._phi_list.append(phi)

    @property
    def previous_phi(self):
        try:
            return self._phi_list[-2]
        except IndexError:
            return None

    @property
    def delta_phi(self):
        try:
            return self._phi_list[-1] - self._phi_list[-2]
        except IndexError:
            return None

    def undo_phi_set(self):
        self._phi_list = self._phi_list[:-1]

    @property
    def proposal(self):
        return self._proposal

    @abc.abstractmethod
    def logpdf(self, inputs, log_like, log_prior):
        return None

    @abc.abstractmethod
    def inc_log_weights(self, inputs, log_like, log_prior, delta_phi):
        return None

    @staticmethod
    def _log_prob_sum(x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = x.sum(axis=1, keepdims=True)
            # probability 0/0 => 0, set nan to -inf
            np.copyto(y, -np.inf, where=np.isnan(y))
        return y


class GeometricPath(PathBase):
    def __init__(self, proposal=None, required_phi=1):
        super().__init__(proposal)
        self._lambda = None
        self.required_phi_list = required_phi

    @property
    def required_phi_list(self):
        return self._required_phi_list.copy()

    @required_phi_list.setter
    def required_phi_list(self, phi):
        if isinstance(phi, float) or isinstance(phi, int):
            phi = [phi]
        self._required_phi_list = sorted([p for p in phi if p < 1])
        self._lambda = min(self._required_phi_list + [1])

    def logpdf(self, inputs, log_like, log_prior):
        log_p = self._get_proposal_logpdf(inputs, log_prior)
        # Inline arguments for faster code
        return self._log_prob_sum(self._eval_target(log_like, log_prior, log_p, self.phi))

    def inc_log_weights(self, inputs, log_like, log_prior):
        log_p = self._get_proposal_logpdf(inputs, log_prior)
        args = log_like, log_prior, log_p
        numer = self._eval_target(*args, self.phi)
        denom = self._eval_target(*args, self.previous_phi)
        return self._log_prob_sum(np.hstack((numer, -denom)))

    def _eval_target(self, log_like, log_prior, log_p, phi):
        # Compute exponents
        _lambda = self._lambda
        prior_exp = min(1.0, phi / _lambda)
        prop_exp = max(0.0, (_lambda - phi) / _lambda)

        # Compute weighted log components with optimized logic
        col1 = log_like * phi  # always used
        # Avoid if-else by multiplying by zero if expo == 0 (fastest)
        col2 = log_prior * prior_exp
        col3 = log_p * prop_exp

        # numpy column-stack for speed
        # use np.column_stack instead of np.hstack, it's more appropriate here
        return np.column_stack((col1, col2, col3))

    def _get_proposal_logpdf(self, inputs, log_prior):
        if self._proposal is not None:
            return self._proposal.logpdf(inputs).reshape(-1, 1)
        else:
            return log_prior
