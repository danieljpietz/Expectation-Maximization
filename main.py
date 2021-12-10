import load as ld
import train
from helpers import *
import numpy as np
from matplotlib import pyplot as plt
try:
    import seaborn as sb
    sb.set_theme()
except ModuleNotFoundError:
    pass

synthetic = True

if synthetic:
    # Feature count
    ld.d = 2
    # Data point count
    n = 300
    data, labels = get_synthetic_data(n, ld.d)
else:
    # Load in the MNIST Data, classes are the 2 numbers we want to distinguish
    # So in this case we are looking at ones and zeros
    ld.load_and_encode(classes=[0, 1])
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
    labels_predicted, likelihoods, _, test_accuracy = train.EM(data, params, labels)
else:
    labels_predicted, likelihoods, _ = train.EM(ld.X_train, params)

# Check against true labels
res = labels_predicted == np.squeeze(labels)

# Since we never look at our labels, they can be flipped
# so both 0% and 100% are perfect fits. Thus, we
# map (0, 0.5) and (0.5, 1) -> (0,1)
accuracy = 2 * abs(0.5 - (np.count_nonzero(res) / res.shape[0]))

# Print our accuracy.
print(accuracy)

# Flip our classes for the plot
if (np.count_nonzero(res) / res.shape[0]) < 0.5:
    labels = np.logical_not(labels)

# Plot our results
plt.plot(likelihoods)
plt.title("Log Likelihood vs vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.savefig("Plots/LogLikli.png")
plt.show()


plt.plot(test_accuracy)
plt.title("TestAccuracy vs vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Classification Accuracy")
plt.savefig("Plots/TestAccuracy.png")
plt.show()

if ld.d == 2:

    plot_size = 5

    plt.scatter(data[labels == 1].T[0], data[labels == 1].T[1], s=plot_size)
    plt.scatter(data[labels == 0].T[0], data[labels == 0].T[1], s=plot_size)
    plt.title("Ground Truth")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend(["Class 1", "Class 2"])
    plt.savefig("Plots/GroundTruth.png")
    plt.show()

    fig = plt.figure()
    plt.scatter(data[labels_predicted == 1].T[0], data[labels_predicted == 1].T[1], s=plot_size)
    plt.scatter(data[labels_predicted == 0].T[0], data[labels_predicted == 0].T[1], s=plot_size)
    plt.title("Predicted Classes")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend(["Class 1", "Class 2"])
    x_range = fig.get_axes()[0].get_xlim()
    y_range = fig.get_axes()[0].get_ylim()
    plt.savefig("Plots/Predicted.png")
    plt.show()



    plt.scatter(data[np.logical_and(labels_predicted == 1, labels == 0)].T[0],
                data[np.logical_and(labels_predicted == 1, labels == 0)].T[1], s=plot_size)
    plt.scatter(data[np.logical_and(labels_predicted == 0, labels == 1)].T[0],
                data[np.logical_and(labels_predicted == 0, labels == 1)].T[1], s=plot_size)
    plt.title("Misclassified Points True Class")
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend(["Class 1", "Class 2"])
    plt.savefig("Plots/Misclass.png")
    plt.show()

elif ld.d == 3:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.scatter(data[labels == 1].T[0], data[labels == 1].T[1], data[labels == 1].T[2])
    plt.scatter(data[labels == 0].T[0], data[labels == 0].T[1], data[labels == 0].T[2])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.scatter(data[labels_predicted == 0].T[0], data[labels_predicted == 0].T[1], data[labels_predicted == 0].T[2])
    plt.scatter(data[labels_predicted == 1].T[0], data[labels_predicted == 1].T[1], data[labels_predicted == 1].T[2])
    plt.show()




