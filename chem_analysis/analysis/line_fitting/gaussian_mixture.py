import numpy as np
import plotly.graph_objs as go
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.stats import norm

# Define a function for a bimodal distribution (sum of two Gaussians)
def bimodal(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    return (amp1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(-(x - mean1)**2 / (2 * sigma1**2)) +
            amp2 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(-(x - mean2)**2 / (2 * sigma2**2)))


def method1(x, y):
    # Fit the data to the bimodal model
    params, covariance = curve_fit(bimodal, x, y, p0=[20, 4.4, 0.2, 15, 5.6, 0.2])

    # Extract the parameters for the two Gaussian distributions
    amp1, mean1, sigma1, amp2, mean2, sigma2 = params

    # Create the fitted curves for the two Gaussians
    fitted_curve1 = amp1 / (sigma1 * np.sqrt(2 * np.pi)) * np.exp(-(x - mean1) ** 2 / (2 * sigma1 ** 2))
    fitted_curve2 = amp2 / (sigma2 * np.sqrt(2 * np.pi)) * np.exp(-(x - mean2) ** 2 / (2 * sigma2 ** 2))

    # Plot the data and the fitted curves
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y))
    fig.add_trace(go.Scatter(x=x, y=fitted_curve1))
    fig.add_trace(go.Scatter(x=x, y=fitted_curve2))
    fig.add_trace(go.Scatter(x=x, y=fitted_curve1 + fitted_curve2))
    fig.write_html("temp1.html", auto_open=True)

    print("Gaussian 1 Parameters:")
    print("Amplitude:", amp1)
    print("Mean:", mean1)
    print("Standard Deviation:", sigma1)

    print("Gaussian 2 Parameters:")
    print("Amplitude:", amp2)
    print("Mean:", mean2)
    print("Standard Deviation:", sigma2)


def _get_peak(fit, x):
    peaks = []
    for i in range(len(fit.means_)):
        mean = fit.means_[i]
        covs = fit.covariances_[i]
        weights = fit.weights_[i]
        print(mean, covs, weights)
        peak = norm.pdf(x, float(mean[0]), np.sqrt(float(covs[0][0]))) * weights
        peaks.append(peak)
    return peaks

def method2(x, y):
    gmm = GaussianMixture(4)
    n= 50_000
    y_ = np.concatenate((np.random.normal(0, 5, n), np.random.normal(10, 2, n),  np.random.normal(12, 0.4, n),
                         np.random.normal(7, 2, n)))
    gmm.fit(y_.reshape(-1, 1))

    x = np.linspace(np.min(y_), np.max(y_), 200)
    peaks = _get_peak(gmm, x)
    fig = go.Figure()
    y, bins = np.histogram(y_, bins=200)
    fig.add_trace(go.Scatter(x=bins[1:], y=y/np.max(y)))
    max_=np.max(np.sum(peaks, axis=0))
    for peak in peaks:
        fig.add_trace(go.Scatter(x=x, y=peak/max_))
    fig.add_trace(go.Scatter(x=x, y=np.sum(peaks, axis=0)/max_))
    fig.write_html("temp2.html", auto_open=True)


def main():
# Sample data points
    x = np.linspace(4, 6, 100)
    y = bimodal(x, 30, 4.5, 0.2, 20, 5.5, 0.2)  # You can replace this with your actual data
    y_noisy = y + 0.5 * np.random.normal(size=len(x))

    # method1(x, y_noisy)
    method2(x, y_noisy)


if __name__ == "__main__":
    main()
