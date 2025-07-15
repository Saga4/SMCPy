import numpy as np

from .log_likelihoods import BaseLogLike


class ApproxHierarch(BaseLogLike):
    """
    Adapted from:
        - Wu et al. ASCE-ASME J Risk and Uncert in Engrg Sys 2019
        - Economides et al. Hierarchical Bayesian UQ for a Model of the Red
          Blood Cell, Phys Rev App 2021
        - Talk by Lucas Amoudruz at SIAM UQ 22 “Inference of the Erythrocytes
          Mechanical Properties Through a Hierarchical Bayesian Framework”

    """

    def __init__(self, model, data, args):
        """
        :params model: user implementation of conditional probabilities p(x|z)
                       where x are samples from a known random effect posterior
                       and z is the proposed vector of overall effect parameters
        :type model:   class
        :params data:  collection of posterior samples from random effects
        :type data:    3D-array-like (number rand effs, number samples,
                       number overall effect parameters)
        :params args:  args[0] is the rand eff marginal log likelihoods and
                       args[1] is the log prior probabilities of the posterior
                       samples provided in the data array
        :type args:    list-like; args[0] is list-like, args[1] is 2D-array
                       with shape (number rand effs, number posterior samples)
        """
        super().__init__(model, data, args)

    def __call__(self, inputs):
        self._override_model_wrapper(inputs)
        # Preallocate output likelihoods shape: (n_inputs, n_random_effects)
        n_inputs = inputs.shape[0]
        n_rand_effs = len(self._data)
        log_like = np.full((n_inputs, n_rand_effs), -np.inf)
        mlls = self._args[0]      # marginal log-likelihoods (list-like)
        log_priors = self._args[1] # shape: (n_rand_effs, n_posterior_samples)

        for i, (d, log_prior, mll) in enumerate(zip(self._data, log_priors, mlls)):
            # Vectorized evaluation for each random effect
            # _get_output(d): d.shape = (n_samples, params)
            log_conditionals = self._get_output(d)
            # log_prior: shape (n_samples,), log_conditionals: (n_inputs, n_samples)
            # Broadcasting log_conditionals - log_prior
            vals = log_conditionals - log_prior
            # logsumexp along samples axis (axis=1): much faster and more stable numerically
            ll = mll - np.log(d.shape[0]) + self._logsumexp(vals)
            log_like[:, i] = ll

        # Avoid reshaping if not required, just ensure 2D column vector output
        return log_like.sum(axis=1)[:, None]

    def _override_model_wrapper(self, inputs):
        # NOTE: Caches the model relevant to inputs, saving a per-sample call
        model = self._model(inputs)
        self.set_model_wrapper(lambda dummy, x: model(x))

    @staticmethod
    def _logsum(Z):
        """
        Assumes summing over columns.
        """
        Z = -np.sort(-np.array(Z), axis=1)  # descending over columns
        Z0 = Z[:, [0]]
        Z_shifted = Z[:, 1:] - Z0
        return Z0.flatten() + np.log(1 + np.sum(np.exp(Z_shifted), axis=1))

    @staticmethod
    def _logsumexp(Z):
        """
        Numerically stable logsumexp along columns (axis=1)
        Equivalent to log(sum(exp(Z), axis=1)), but stable for large/small numbers.
        """
        Z = np.asarray(Z)
        # Z: (n_inputs, n_samples)
        maxZ = np.max(Z, axis=1, keepdims=True)
        sumexp = np.exp(Z - maxZ).sum(axis=1)
        return (maxZ[:, 0] + np.log(sumexp))


class MVNHierarchModel:
    def __init__(self, inputs):
        self._inputs, self._hyperparams = self._separate_inputs(inputs)
        self._cov = self.build_cov_array()
        self._term1, self._term2, self._VI = self._compute_likelihood_consts()

    def __call__(self, data):
        term3 = self._compute_squared_mahalanobis(data)
        return -1 / 2 * (self._term1 + self._term2 + term3)

    def build_cov_array(self):
        dim = self._inputs.shape[1]
        cov = np.zeros((self._inputs.shape[0], dim, dim))
        i, j = np.triu_indices(dim)
        cov[:, i, j] = self._hyperparams
        cov[:, j, i] = self._hyperparams
        return cov

    def _compute_squared_mahalanobis(self, data):
        delta = data - np.expand_dims(self._inputs, 1)
        return np.einsum("inj,ijk,ink->in", delta, self._VI, delta)

    def _separate_inputs(self, inputs):
        p = int(-3 / 2 + np.sqrt(9 / 4 + 2 * inputs.shape[1]))
        self._check_if_dimension_valid(p, inputs.shape[1] - p)
        return inputs[:, :p], inputs[:, p:]

    def _compute_likelihood_consts(self):
        r, c = self._inputs.shape
        term1 = np.full((r, 1), c * np.log(2 * np.pi))
        term2 = np.linalg.slogdet(self._cov)[1].reshape(-1, 1)
        VI = np.linalg.inv(self._cov)
        return term1, term2, VI

    @staticmethod
    def _check_if_dimension_valid(p, c):
        if p * (p + 1) / 2 - c != 0:
            raise IndexError(
                "Provided numbers of parameters and covariance " "terms are invalid."
            )
