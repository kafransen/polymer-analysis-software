# PyMCR uses the Logging facility to capture messaging
# Sends logging messages to stdout (prints them)
import sys


M = 21
N = 21
P = 101
n_components = 2

C_img = _np.zeros((M, N, n_components))
C_img[..., 0] = _np.dot(_np.ones((M, 1)), _np.linspace(0, 1, N)[None, :])
C_img[..., 1] = 1 - C_img[..., 0]

St_known = _np.zeros((n_components, P))
St_known[0, 40:60] = 1
St_known[1, 60:80] = 2

C_known = C_img.reshape((-1, n_components))

D_known = _np.dot(C_known, St_known)

mcrar = McrAR()
mcrar.fit(D_known, ST=St_known, verbose=True)
# assert_equal(1, mcrar.n_iter_opt)
assert ((mcrar.D_ - D_known) ** 2).mean() < 1e-10
assert ((mcrar.D_opt_ - D_known) ** 2).mean() < 1e-10

mcrar = McrAR()
mcrar.fit(D_known, C=C_known)
# assert_equal(1, mcrar.n_iter_opt)
assert ((mcrar.D_ - D_known) ** 2).mean() < 1e-10
assert ((mcrar.D_opt_ - D_known) ** 2).mean() < 1e-10