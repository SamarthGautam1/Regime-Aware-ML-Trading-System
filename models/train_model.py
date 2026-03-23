import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from utils.helper import setup_logger

logger = setup_logger()


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Train a Logistic Regression model inside a sklearn Pipeline.

    The Pipeline does two things in sequence:
        Step 1 — StandardScaler: scales each feature to mean=0, std=1.
                 This is necessary because features like 'zscore_20d' and
                 'return_1d' live on very different numerical scales.
                 Logistic Regression is sensitive to scale.

        Step 2 — LogisticRegression: learns weights for each feature
                 to predict the probability of an upward move tomorrow.

    Why a Pipeline?
        When we later call model.predict_proba(X_test), the pipeline
        automatically applies the SAME scaling it learned from training data.
        This is how we prevent data leakage — the scaler never sees test data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training. Must use get_feature_columns() to select.
    y_train : pd.Series
        Binary target (1 = up day tomorrow, 0 = down day tomorrow).

    Returns
    -------
    Pipeline
        Fitted sklearn pipeline (scaler + model). Ready for prediction.
    """

    pipeline = Pipeline([
        # 'scaler' is just a name we give this step — can be anything
        ("scaler", StandardScaler()),

        # max_iter=1000 — default 100 sometimes fails to converge,
        # bumping it up prevents warnings without changing behavior.
        # random_state=42 — ensures reproducibility.
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    # .fit() does two things internally:
    #   1. Fits the scaler on X_train (learns mean + std of each feature)
    #   2. Scales X_train and fits logistic regression on the scaled data
    pipeline.fit(X_train, y_train)

    logger.info("Model trained on %d samples, %d features.", len(X_train), X_train.shape[1])

    return pipeline
