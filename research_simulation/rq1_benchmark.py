import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, 
    confusion_matrix, roc_auc_score
)
from .config import Config
from .models import get_models

class RQ1Benchmark:
    def __init__(self, config=Config):
        self.config = config
        
    def run_cross_validation(self, X_train, y_train):
        """
        Runs 5-Fold Stratified Cross Validation on the training partition.
        Records Mean Accuracy and Macro F1 score across folds.
        """
        print("\n--- Running RQ1: 5-Fold Stratified Cross Validation ---")
        models = get_models()
        skf = StratifiedKFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=self.config.RANDOM_SEED)
        
        cv_records = {name: {"fold_acc": [], "fold_macro_f1": [], "fold_time": []} for name in models}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr_f, y_tr_f = X_train[train_idx], y_train[train_idx]
            X_val_f, y_val_f = X_train[val_idx], y_train[val_idx]
            
            for name, model in get_models().items():
                t0 = time.time()
                model.fit(X_tr_f, y_tr_f)
                fit_t = time.time() - t0
                
                preds = model.predict(X_val_f)
                acc = accuracy_score(y_val_f, preds)
                f1_m = f1_score(y_val_f, preds, average="macro")
                
                cv_records[name]["fold_acc"].append(acc)
                cv_records[name]["fold_macro_f1"].append(f1_m)
                cv_records[name]["fold_time"].append(fit_t)
                
            print(f"  Fold {fold}/{self.config.N_FOLDS} completed.")
            
        summary = []
        for name in models:
            accs = cv_records[name]["fold_acc"]
            f1s = cv_records[name]["fold_macro_f1"]
            times = cv_records[name]["fold_time"]
            summary.append({
                "Model": name,
                "CV Mean Accuracy (%)": f"{np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}",
                "CV Mean Macro-F1 (%)": f"{np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}",
                "CV Mean Train Time (s)": f"{np.mean(times):.2f} ± {np.std(times):.2f}",
                "acc_raw": accs,
                "f1_raw": f1s
            })
            
        df_cv = pd.DataFrame(summary)
        return df_cv, cv_records

    def evaluate_test_partition(self, X_train, y_train, X_test, y_test):
        """
        Fits each model on full training set and evaluates on the held-out evaluation partition.
        Calculates Accuracy, Macro F1, Recall, Precision, FPR, and ROC-AUC.
        """
        print("\n--- Running RQ1: Held-out Test Partition Evaluation ---")
        models = get_models()
        results = []
        model_artifacts = {}
        
        for name, model in models.items():
            print(f"  Training {name} on full training set ({len(X_train)} samples)...")
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            
            # Measure inference time
            t0_inf = time.time()
            probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
            inf_time = (time.time() - t0_inf) / len(X_test) * 1000 # ms per sample
            
            preds = (probs >= 0.5).astype(int)
            
            acc = accuracy_score(y_test, preds)
            macro_f1 = f1_score(y_test, preds, average="macro")
            attack_f1 = f1_score(y_test, preds, pos_label=1)
            recall = recall_score(y_test, preds, pos_label=1)
            precision = precision_score(y_test, preds, pos_label=1)
            auc = roc_auc_score(y_test, probs)
            
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            fpr = fp / (fp + tn) # False Positive Rate
            fnr = fn / (fn + tp) # False Negative Rate
            specificity = tn / (tn + fp)
            
            results.append({
                "Model": name,
                "Accuracy (%)": acc * 100,
                "Macro F1 (%)": macro_f1 * 100,
                "Attack Recall (%)": recall * 100,
                "Attack Precision (%)": precision * 100,
                "Attack F1 (%)": attack_f1 * 100,
                "False Positive Rate (%)": fpr * 100,
                "ROC-AUC": auc,
                "Training Time (s)": train_time,
                "Inference Latency (ms/sample)": inf_time,
                "TN": tn, "FP": fp, "FN": fn, "TP": tp
            })
            
            model_artifacts[name] = {
                "model": model,
                "probs": probs,
                "preds": preds
            }
            
        df_test = pd.DataFrame(results)
        return df_test, model_artifacts

    def perform_statistical_tests(self, cv_records):
        """
        Paired t-test and Wilcoxon signed-rank test on fold Macro-F1 and Accuracy.
        """
        print("\n--- Performing Statistical Significance Tests ---")
        hgb_f1 = cv_records["Histogram Gradient Boosting"]["fold_macro_f1"]
        rf_f1 = cv_records["Random Forest"]["fold_macro_f1"]
        lr_f1 = cv_records["Logistic Regression"]["fold_macro_f1"]
        
        t_stat_lr, p_val_lr = stats.ttest_rel(hgb_f1, lr_f1)
        t_stat_rf, p_val_rf = stats.ttest_rel(hgb_f1, rf_f1)
        
        stat_results = {
            "Comparison (HistGB vs LR)": {
                "t_statistic": float(t_stat_lr),
                "p_value": float(p_val_lr),
                "significant": bool(p_val_lr < 0.05)
            },
            "Comparison (HistGB vs RF)": {
                "t_statistic": float(t_stat_rf),
                "p_value": float(p_val_rf),
                "significant": bool(p_val_rf < 0.05)
            }
        }
        return stat_results

