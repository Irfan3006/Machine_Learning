import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from .config import Config

class DataPipeline:
    def __init__(self, config=Config):
        self.config = config
        self.preprocessor = None
        self.feature_names = []
        
    def load_data(self):
        print(f"Loading data from {self.config.TRAIN_PATH} and {self.config.TEST_PATH}...")
        df_train = pd.read_csv(self.config.TRAIN_PATH)
        df_test = pd.read_csv(self.config.TEST_PATH)
        
        X_train_raw = df_train.drop(columns=self.config.DROP_COLS)
        y_train = df_train[self.config.TARGET_COL].values
        
        X_test_raw = df_test.drop(columns=self.config.DROP_COLS)
        y_test = df_test[self.config.TARGET_COL].values
        
        num_cols = [c for c in X_train_raw.columns if c not in self.config.CATEGORICAL_COLS]
        self.feature_names = num_cols + self.config.CATEGORICAL_COLS
        
        return X_train_raw, y_train, X_test_raw, y_test, num_cols
        
    def fit_transform(self, X_train_raw, y_train, X_test_raw, num_cols):
        cv = KFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.RANDOM_SEED)
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', TargetEncoder(cv=cv, smooth="auto"), self.config.CATEGORICAL_COLS)
            ]
        )
        
        X_train_proc = self.preprocessor.fit_transform(X_train_raw, y_train)
        X_test_proc = self.preprocessor.transform(X_test_raw)
        
        return X_train_proc, X_test_proc

