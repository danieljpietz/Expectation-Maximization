import load as ld
import train
from helpers import *

# Load in the MNIST Data, classes are the 2 numbers we want to distinguish
# So in this case we are looking at ones and zeros
ld.load_and_encode(classes=[0, 1])

# The SK Learn EM Algorithm for a baseline
# model = GaussianMixture(n_components=2, max_iter=1000)
# model.fit(ld.X_train)

synthetic = True

if synthetic:
    # Feature count
    ld.d = 15
    # Data point count
    n = 12000
    data, labels = get_synthetic_data(n, ld.d)
else:
    data = ld.X_train
    labels = np.squeeze(ld.labels_binary_train)

# Initialize random GMM parameters
phi1 = np.random.rand(1, 1)
mean1 = np.squeeze(np.random.rand(1, ld.d))
mean2 = np.squeeze(np.random.rand(1, ld.d))
cov1 = np.eye(ld.d)
cov2 = np.eye(ld.d)

# Create our parameter vector
params = [phi1, mean1, mean2, cov1, cov2]

if synthetic:
    forecasts, _, _, _ = train.EM(data, params)
else:
    forecasts, _, _, _ = train.EM(ld.X_train, params)

# Check against true labels
res = forecasts == np.squeeze(labels)

# Since we never look at our labels, they can be flipped
# so both 0% and 100% are perfect fits. Thus, we
# map (0, 0.5) and (0.5, 1) -> (0,1)
accuracy = 2 * abs(0.5 - (np.count_nonzero(res) / res.shape[0]))

# Print our accuracy.
print(accuracy)
