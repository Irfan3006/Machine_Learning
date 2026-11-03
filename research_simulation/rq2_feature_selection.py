import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from .config import Config

class RQ2FeatureSelection:
    def __init__(self, config=Config):
        self.config = config
        
    def rank_features(self, X_train, y_train, feature_names):
        """
        Computes feature importance ranking using Random Forest.
        """
        print("\n--- Running RQ2: Feature Importance Ranking ---")
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=self.config.RANDOM_SEED)
        rf.fit(X_train, y_train)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        ranking_df = pd.DataFrame({
            "Rank": list(range(1, len(feature_names) + 1)),
            "Feature": [feature_names[i] for i in indices],
            "Importance": [importances[i] for i in indices],
            "Index": indices
        })
        return ranking_df, indices

    def evaluate_subsets(self, X_train, y_train, X_test, y_test, ranking_indices):
        """
        Evaluates model performance across different feature counts: [42, 30, 20, 15, 10, 5].
        Measures Macro-F1, Accuracy, Recall, FPR, and Training Time (computational cost).
        """
        print("\n--- Running RQ2: Evaluating Feature Subsets on HistGB ---")
        subsets = self.config.FEATURE_SUBSETS
        results = []
        
        # Measure baseline full features first
        baseline_f1 = None
        baseline_time = None
        
        for k in subsets:
            selected_idx = ranking_indices[:k]
            X_tr_sub = X_train[:, selected_idx]
            X_te_sub = X_test[:, selected_idx]
            
            clf = HistGradientBoostingClassifier(random_state=self.config.RANDOM_SEED)
            
            t0 = time.time()
            clf.fit(X_tr_sub, y_train)
            train_time = time.time() - t0
            
            preds = clf.predict(X_te_sub)
            acc = accuracy_score(y_test, preds)
            macro_f1 = f1_score(y_test, preds, average="macro")
            recall = recall_score(y_test, preds, pos_label=1)
            
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            fpr = fp / (fp + tn)
            
            if k == subsets[0]:
                baseline_f1 = macro_f1
                baseline_time = train_time
                
            f1_retention = (macro_f1 / baseline_f1) * 100
            time_reduction = ((baseline_time - train_time) / baseline_time) * 100 if baseline_time > 0 else 0
            
            print(f"  Top-{k:02d} Features: Macro-F1={macro_f1*100:.2f}%, TrainTime={train_time:.2f}s (Time Saved: {time_reduction:.1f}%, F1 Retained: {f1_retention:.1f}%)")
            
            results.append({
                "Num_Features": k,
                "Accuracy (%)": acc * 100,
                "Macro_F1 (%)": macro_f1 * 100,
                "Attack_Recall (%)": recall * 100,
                "FPR (%)": fpr * 100,
                "Training_Time_s": train_time,
                "F1_Retention (%)": f1_retention,
                "Time_Reduction (%)": max(0.0, time_reduction)
            })
            
        df_subsets = pd.DataFrame(results)
        return df_subsets

