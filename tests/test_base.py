
import numpy as np


def generate_signal():
    from scipy.stats import norm
    nx = 1000
    ny = 3
    x = np.linspace(0, nx, nx)
    y = np.empty((ny, nx))
    for i in range(ny):
        rv = norm(loc=nx * np.random.random(1), scale=10)
        y[i, :] = np.linspace(0, nx, nx) + 20 * np.random.random(nx) * np.random.random(1) + \
                  5000 * rv.pdf(x) * np.random.random(1)

    return x,y


def local_run():
    df = pd.DataFrame(data=y.T, index=x)
    df.columns = ["RI", "UV", "LS"]
    df.index.names = ["time"]
    chro = Chromatogram(df)
    chro.baseline(deg=1)
    chro.auto_peak_picking()
    chro.plot()
    chro.stats()
    print("done")


if __name__ == "__main__":
    local_run()