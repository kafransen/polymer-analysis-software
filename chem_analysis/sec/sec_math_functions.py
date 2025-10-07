import numpy as np

from chem_analysis.utils.math import check_for_flip


def calculate_Mn_D_from_wi(mw_i: np.ndarray, wi: np.ndarray) -> tuple[float, float]:
    """ calculate Mn and D from wi vs MW data (MW goes low to high) """
    # flip data if giving backwards; it should be MW low to MW high
    mw_i, wi = check_for_flip(mw_i, wi)

    data_points = len(mw_i)
    wi_d_mi = np.zeros(data_points)
    wi_m_mi = np.zeros(data_points)
    for i in range(data_points):
        if mw_i[i] != 0:
            wi_d_mi[i] = wi[i] / mw_i[i]
        wi_m_mi[i] = wi[i] * mw_i[i]

    mw_n = np.sum(wi) / np.sum(wi_d_mi)
    mw_w = np.sum(wi_m_mi) / np.sum(wi)
    mw_d = mw_w / mw_n
    return mw_n, mw_d
