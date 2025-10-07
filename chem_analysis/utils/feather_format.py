
import pathlib
from typing import Sequence

import numpy as np
import pyarrow as pa

from chem_analysis.utils.math import unpack_time_series


def numpy_to_feather(array_: np.ndarray, file_path: str | pathlib.Path, headers: Sequence[str] = None):
    """
    Save numpy array to feather file

    Parameters
    ----------
    array_:
        numpy array
    file_path:
        file path
    """
    if not isinstance(file_path, pathlib.Path):
        file_path = pathlib.Path(file_path)
    if file_path.suffix != ".feather":
        file_path = file_path.with_suffix(".feather")
    if headers is None:
        headers = [str(i) for i in range(array_.shape[0])]
    else:
        if len(headers) != array_.shape[0]:
            raise ValueError(f"header need to be the same length as array_.shape[1]."
                             f"\n\tExpected: {array_.shape[1]}"
                             f"\n\tGiven: {len(headers)}"
                             )

    arrays = [pa.array(col) for col in array_]
    batch = pa.RecordBatch.from_arrays(arrays, names=headers)
    with pa.OSFile(str(file_path), 'wb') as sink:
        with pa.RecordBatchStreamWriter(sink, batch.schema) as writer:
            writer.write_batch(batch)


def feather_to_numpy(file_path: str | pathlib.Path) -> tuple[np.ndarray, list[str]]:
    """

    Parameters
    ----------
    file_path:
        feather file

    Returns
    -------

    """
    if isinstance(file_path, pathlib.Path):
        file_path = str(file_path)

    source = pa.memory_map(file_path, 'r')
    table = pa.ipc.RecordBatchStreamReader(source).read_all()
    data = np.empty(np.flip(table.shape))
    for i, col in enumerate(table):
        data[i, :] = col.to_numpy()
    return data, table.column_names


def unpack_and_merge_time_series_feather_files(paths: Sequence[str | pathlib.Path]) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(paths) == 0:
        raise ValueError("No files to unpack.")

    z_data = []
    time_data = []
    for path in paths:
        x, time_, z = unpack_time_series(feather_to_numpy(path))
        z_data.append(z)
        time_data.append(time_)

    return x, np.concatenate(time_data), np.concatenate(z_data)
