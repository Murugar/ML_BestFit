from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve, ShuffleSplit


# Test Function
test_function = lambda X: np.sin(100 + np.pi * X)

# Generate random observations around the test function
n_samples = 100
degrees = [1, 10, 20, 100, 500, 1000, 10000] 

X = np.linspace(-1, 1, n_samples)
y = test_function(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(25, 18))

models = {}

# Plot all machine learning algorithm models
for idx, degree in enumerate(degrees):
    ax = plt.subplot(1, len(degrees), idx + 1)
    plt.setp(ax, xticks=(), yticks=())
    
    # Define the model
    polynomial_features = PolynomialFeatures(degree=degree)
    model = make_pipeline(polynomial_features, LinearRegression())
    
    models[degree] = model
    
    # Train the model
    model.fit(X[:, np.newaxis], y)
    
    # Evaluate the model using cross-validation
    scores = cross_val_score(model, X[:, np.newaxis], y)
    
    X_test = X
    plt.plot(X_test, model.predict(X_test[:, np.newaxis]), label="Model")
    plt.scatter(X, y, edgecolor='b', s=30, label="Observations")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim((-3, 3))
    
    plt.title("Degree {}\nMSE = {:.2e}".format(degree, -scores.mean()))

plt.show()
