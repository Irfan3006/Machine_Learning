from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from .config import Config

def get_models(class_weight=None, seed=Config.RANDOM_SEED):
    """
    Returns the three candidate classifiers configured for the experiment.
    - Logistic Regression (Linear baseline)
    - Random Forest (Bagging ensemble, non-linear)
    - Histogram-based Gradient Boosting (Boosting ensemble, non-linear, optimized for tabular data)
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, 
            class_weight=class_weight, 
            random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=15, 
            class_weight=class_weight, 
            n_jobs=-1, 
            random_state=seed
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            class_weight=class_weight, 
            random_state=seed
        )
    }

