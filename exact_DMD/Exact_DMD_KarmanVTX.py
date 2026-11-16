import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the data from the .mat file
path = 'add path to Vortall'
data = loadmat(path)
Vortall = data['VORTALL']

print("Data shape:", Vortall.shape)

# Make the data matrices 
X = Vortall[:, :-1]  # All columns except the last one
Y = Vortall[:, 1:]  # All columns except the first one

# compute the SVD of X
U, S, Vh = np.linalg.svd(X, full_matrices=False)
r = 6
U_r = U[:, :r]
S_r = np.diag(S[:r])
Vh_r = Vh[:r, :].T
Sigma_inv = np.diag(1 / S[:r])

# Solve the small eigenvalue problem
B = Y @ Vh_r @ Sigma_inv
A_tilde = U_r.T @ B
eigenvalues, W = np.linalg.eig(A_tilde)
Phi = B @ W

print("Eigenvectors:", W.shape)
print("Eigenvalues:",eigenvalues.shape)
print("Phi shape:", Phi.shape)


# fitting the initial condition
xo = Vortall[:, 0]
b = np.linalg.lstsq(Phi, xo, rcond=None)[0]
k = 80
x_dmd = (Phi @ (b* eigenvalues**k)).real

#  Reshape the snapshot vectors into images and plot them
grid_shape = (199, 449)
actual = Vortall[:, k].reshape(grid_shape, order="F")
reconstructed = x_dmd.reshape(grid_shape, order="F")
limit = np.max(np.abs(actual))

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, field, title in zip(
    axes, [actual, reconstructed], ["Original vorticity", "Exact DMD"]
):
    im = ax.imshow(field, cmap="RdBu_r", origin="lower",
                   vmin=-limit, vmax=limit)
    ax.set_title(f"{title} · snapshot {k}")

fig.colorbar(im, ax=axes, label="Vorticity")
plt.show()
