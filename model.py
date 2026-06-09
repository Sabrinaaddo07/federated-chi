"""
Model wrapper for scikit-learn Logistic Regression to work with Flower.
Flower expects models to have get_parameters() and set_parameters() methods.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def create_model():
    """
    Create a Logistic Regression model configured for federated learning.
    - max_iter=1: Train for 1 epoch per fit() call (local epoch)
    - warm_start=True: Continue training from previous coefficients
    - solver='lbfgs': Works well for multinomial classification
    - multi_class='multinomial': For 10-class MNIST classification
    """
    return LogisticRegression(
        max_iter=1,
        warm_start=True,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
    )


def get_parameters(model):
    """
    Extract model parameters as a list of NumPy arrays.
    Flower sends these to the server for aggregation.
    Returns: [coef_, intercept_] - both are NumPy arrays
    """
    if model.coef_ is None:
        # Model not fitted yet - return zeros with correct shape
        # MNIST has 10 classes, 784 features (28x28 images)
        return [
            np.zeros((10, 784), dtype=np.float64),  # coef_
            np.zeros(10, dtype=np.float64),          # intercept_
        ]
    return [model.coef_, model.intercept_]


def set_parameters(model, parameters):
    """
    Load parameters into the model.
    Called when receiving aggregated parameters from the server.
    parameters: [coef_, intercept_] as list of NumPy arrays
    """
    model.coef_ = parameters[0]
    model.intercept_ = parameters[1]
    # Mark model as fitted so predict() works
    model.classes_ = np.arange(10)