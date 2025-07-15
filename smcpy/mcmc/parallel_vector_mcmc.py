import numpy as np

from .vector_mcmc import VectorMCMC
from ..log_likelihoods import Normal


class ParallelVectorMCMC(VectorMCMC):
    """
    Enables use of MPI to split model evaluations over a distributed memory
    system.

    ParallelVectorMCMC is set up such that all positive ranks have effectively
    garbage values. These garbage values serve the purpose of allowing the
    MCMCBase class, which was not written with MPI in mind, to function
    normally and propagate objects of the right shape and type. At the end of
    an analysis that uses the ParallelVectorMCMC class, outputs on positive ranks
    should be discarded and ONLY the output from rank 0 should be used.
    """
    def __init__(
        self, model, data, priors, mpi_comm, log_like_args=None, log_like_func=Normal
    ):
        self._comm = mpi_comm
        self._size = mpi_comm.Get_size()
        self._rank = mpi_comm.Get_rank()

        super().__init__(model, data, priors, log_like_args, log_like_func)

        self._log_like_func.set_model_wrapper(self.model_wrapper)

        # Determine the output dimension in a safer and faster way
        data_shape = getattr(self._data, "shape", None)
        self._output_dim = data_shape[1] if data_shape is not None and len(data_shape) > 1 else np.size(self._data)

    def model_wrapper(self, model, inputs):
        num_inputs = len(inputs) if hasattr(inputs, '__len__') else np.shape(inputs)[0]
        # Fast split via slicing if possible, else fallback
        if isinstance(inputs, np.ndarray):
            # Handle only 2D arrays or 1D arrays
            if inputs.ndim == 2 and num_inputs % self._size == 0:
                # block slice: same size per process
                block_size = num_inputs // self._size
                start = self._rank * block_size
                end = (self._rank + 1) * block_size
                scattered_inputs = inputs[start:end]
            else:
                partitioned_inputs = np.array_split(inputs, self._size)
                scattered_inputs = self._comm.scatter(partitioned_inputs, root=0)
        else:
            partitioned_inputs = np.array_split(inputs, self._size)
            scattered_inputs = self._comm.scatter(partitioned_inputs, root=0)

        # Compute outputs for this partition if we have work
        if hasattr(scattered_inputs, 'shape') and scattered_inputs.shape[0] > 0:
            scattered_outputs = model(scattered_inputs)
        else:
            # Allocate minimal empty array with correct shape/type
            scattered_outputs = np.empty((0, self._output_dim), dtype=self._data.dtype)

        # Gather outputs across all processes
        gathered_outputs = self._comm.allgather(scattered_outputs)
        # The outputs might be empty; np.concatenate is fine if list is not empty
        if len(gathered_outputs) == 1:
            outputs = gathered_outputs[0]
        else:
            # Use concatenate, since outputs may be variable-length
            outputs = np.concatenate(gathered_outputs, axis=0)
        return outputs
