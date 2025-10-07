
import numpy as np

import chem_analysis.sec as ca_sec


def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    retention_time = data[:, 0]
    data = data[:, 1::2]
    return retention_time, data


def main():
    cal_RI = ca_sec.ConventionalCalibration(lambda time: 10 ** (-0.6 * time + 10.644),
                                            mw_bounds=(160, 1_090_000), name="RI calibration")

    path = r"G:\Other computers\My Laptop\post_doc_2022\Data\polymerizations\DW2-6\DW2_6_SEC.csv"
    retention_time, data = load_data(path)
    signals = []
    for i in range(data.shape[1]):
        signals.append(
            ca_sec.SECSignal(x=retention_time, y=data[:, i], calibration=cal_RI, type_=ca_sec.SECTypes.RI)
        )

    peaks = []
    for sig in signals:
        peaks.append(
            chem_analysis.analysis.peak_picking.scipy_find_peaks(sig)
        )


if __name__ == "__main__":
    main()

