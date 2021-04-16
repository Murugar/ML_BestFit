from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import learning_curve, ShuffleSplit


test_function = lambda X: np.sin(100 + np.pi * X)

# Generate random observations around the ground truth function
n_samples = 100
degrees = [1, 20, 100, 1000, 10000] 

X = np.linspace(-1, 1, n_samples)
y = test_function(X) + np.random.randn(n_samples) * 0.1



models = {}

# Plot all machine learning algorithm models
for idx, degree in enumerate(degrees):
   
    
    # Define the model
    polynomial_features = PolynomialFeatures(degree=degree)
    model = make_pipeline(polynomial_features, LinearRegression())
    
    models[degree] = model
    
   

# Plot learning curves
plt.figure(figsize=(25, 18))

for idx, degree in enumerate(models):
    ax = plt.subplot(1, len(degrees), idx + 1)
    
    plt.title("Degree {}".format(degree))
    plt.grid()
    
    plt.xlabel("Training")
    plt.ylabel("Score")
    
    train_sizes = np.linspace(.6, 1.0, 6)
    
    # Cross-validation with 100 iterations to get a smoother mean test and training
    # score curves, each time with 10% of the data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    
    model = models[degree]
    train_sizes, train_scores, test_scores = learning_curve(
        model, X[:, np.newaxis], y, cv=cv, train_sizes=train_sizes, n_jobs=4)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test")
    
    plt.legend(loc = "best")

plt.show()
