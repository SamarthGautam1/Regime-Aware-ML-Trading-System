import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


def predict_proba(model: Pipeline, X_test: pd.DataFrame) -> pd.Series:
    """
    Generate probability of an upward move for each row in X_test.

    The pipeline internally:
        1. Scales X_test using the scaler fitted on training data
        2. Passes scaled features through logistic regression
        3. Returns class probabilities

    We extract the probability for class 1 (up day) — column index [1].

    Parameters
    ----------
    model : Pipeline
        Fitted pipeline returned by train_model().
    X_test : pd.DataFrame
        Feature matrix for the test period. Same columns as X_train.

    Returns
    -------
    pd.Series
        Probability of upward move for each day. Index matches X_test.
        Values are floats between 0 and 1.
    """

    # predict_proba returns a 2D array: [[prob_down, prob_up], ...]
    # We want the second column — probability of class 1 (up)
    proba = model.predict_proba(X_test)[:, 1]

    return pd.Series(proba, index=X_test.index, name="pred_proba")
