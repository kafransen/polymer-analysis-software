import numpy as np

import chem_analysis.sec as ca_sec


def quadratic(a, b, c):
    return -b + np.sqrt(b ** 2 - 4 * a * c) / (2 * a)


def create_data(Mn: float = 15_000, D: float = 1.05, n: int = 1000):
    mw_i = np.linespace(0, 30, n)
    weight_fraction = 1 / (Mn * np.sqrt(2 * np.pi * np.log(D))) * \
                      np.exp(-1 * (np.log(mw_i / Mn) + np.log(D) / 2) ** 2 / (2 * np.log(D)))
    retention_time = quadratic(0.0167, - 0.9225, 14.087 - np.log10(mw_i))
    return retention_time, weight_fraction


def main():
    retention_time, weight_fraction = create_data()

    def cal_func(time: np.ndarray) -> np.ndarray:
        return 10 ** (0.0167 * time ** 2 - 0.9225 * time + 14.087)

    calibration = ca_sec.ConventionalCalibration(cal_func, lower_bound_mw=900, upper_bound_mw=319_000)

    signal = ca_sec.SECSignal(x=retention_time, y=weight_fraction, calibration=calibration)
    signal.processor.add(chem_analysis.processing.baseline_correction.Polynomial(degree=3))
    peaks = chem_analysis.analysis.peak_picking.find_peaks(signal, lower_bound, upper_bound, )
    signal.print_stats()
    ca_sec.plot(signal)


if __name__ == "__main__":
    main()
