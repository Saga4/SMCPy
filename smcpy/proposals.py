import numpy as np


class MultivarIndependent:
    def __init__(self, *args):
        """
        Helper class to combine arbitrary number of independent proposal
        distributions together into a single object with rvs() and logpdf()
        methods. Intended for use with the smcpy.paths module.

        :param args: arbitrary number of distributions
        :type args: scipy.stats-like distribution objects
        """
        self._dist_list = args
        self._dims = self._get_dims()
        # Precompute split indices for fast partitioning
        # Store as tuple for slight access optimization
        self._split_indices = tuple(np.cumsum(self._dims)[:-1])

    def rvs(self, num_samples, random_state=None):
        return np.hstack(
            [
                np.c_[d.rvs(num_samples, random_state=random_state)]
                for d in self._dist_list
            ]
        )

    def logpdf(self, inputs):
        iterable = zip(self._dist_list, self._partition_inputs(inputs))
        indv_logpdf = np.hstack([d.logpdf(in_) for d, in_ in iterable])
        return np.sum(indv_logpdf, axis=1, keepdims=True)

    def _get_dims(self):
        return [d.rvs(1).size for d in self._dist_list]

    def _partition_inputs(self, inputs):
        # Use precomputed split indices for fast split
        return np.split(inputs, self._split_indices, axis=1)
