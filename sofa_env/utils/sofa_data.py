import numpy as np
import Sofa
import Sofa.Core


def read_only_array_from_data_container(container: Sofa.Core.DataContainer) -> np.ndarray:
    """Returns a read only numpy array of the array in a DataContainer.

    1. call __getitem__ on the DataContainer, to get a numpy array with the same underlying data.
    2. set the numpy to read only.
    """
    read_only_array = container[:]
    read_only_array.flags.writeable = False
    return read_only_array


def write_array_to_data_container(container: Sofa.Core.DataContainer, array: np.ndarray) -> None:
    """Writes values into array of a DataContainer (instead of assigning one array to the other)

    1. call __getitem__ on the DataContainer, to get a numpy array with the same underlying data.
    2. call __setitem__ on the numpy array -> values will be writte to the same memory, as the underlying data in DataContainer.
    Directly calling __setitem__ on DataContainer does not work, because the bound function does not know how to cast a numpy
    array into the data type sofa uses...
    """

    data = container[:]
    data[:] = array
