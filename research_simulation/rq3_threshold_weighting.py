import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from .config import Config

class RQ3ThresholdWeighting:
    def __init__(self, config=Config):
        self.config = config
        
    def evaluate_class_weights(self, X_train, y_train, X_test, y_test):
        """
        Compares model performance between default unweighted (None) and balanced class weights.
        """
        print("\n--- Running RQ3: Class Weighting Comparison ---")
        weight_configs = ["None", "balanced"]
        records = []
        
        for weight in weight_configs:
            cw = None if weight == "None" else "balanced"
            clf = HistGradientBoostingClassifier(class_weight=cw, random_state=self.config.RANDOM_SEED)
            clf.fit(X_train, y_train)
            
            probs = clf.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)
            
            acc = accuracy_score(y_test, preds)
            rec = recall_score(y_test, preds, pos_label=1)
            f1_m = f1_score(y_test, preds, average="macro")
            
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            fpr = fp / (fp + tn)
            
            records.append({
                "Class_Weight": weight,
                "Accuracy (%)": acc * 100,
                "Attack_Recall (%)": rec * 100,
                "FPR (%)": fpr * 100,
                "Macro_F1 (%)": f1_m * 100,
                "TN": tn, "FP": fp, "FN": fn, "TP": tp
            })
            print(f"  Weight={weight}: Acc={acc*100:.2f}%, Recall={rec*100:.2f}%, FPR={fpr*100:.2f}%, Macro-F1={f1_m*100:.2f}%")
            
        return pd.DataFrame(records)

    def sweep_thresholds(self, model, X_test, y_test, model_name="HistGradientBoosting"):
        """
        Sweeps decision threshold from 0.05 to 0.95.
        Analyzes the trade-off between Attack Recall and FPR.
        Identifies operating region where Accuracy >= 93% and Recall >= 93%.
        """
        print(f"\n--- Running RQ3: Decision Threshold Sweeping for {model_name} ---")
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        
        thresholds = self.config.THRESHOLDS
        results = []
        valid_points = []
        
        for th in thresholds:
            preds = (probs >= th).astype(int)
            acc = accuracy_score(y_test, preds)
            rec = recall_score(y_test, preds, pos_label=1)
            prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
            macro_f1 = f1_score(y_test, preds, average="macro")
            
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            fpr = fp / (fp + tn)
            fnr = fn / (fn + tp)
            
            meets_validation = (acc >= self.config.VALIDATION_TARGET_ACCURACY) and (rec >= self.config.VALIDATION_TARGET_RECALL)
            
            if meets_validation:
                valid_points.append(th)
                
            results.append({
                "Threshold": th,
                "Accuracy (%)": acc * 100,
                "Attack_Recall (%)": rec * 100,
                "Attack_Precision (%)": prec * 100,
                "FPR (%)": fpr * 100,
                "FNR (%)": fnr * 100,
                "Macro_F1 (%)": macro_f1 * 100,
                "Meets_Validation": meets_validation,
                "TN": tn, "FP": fp, "FN": fn, "TP": tp
            })
            
        df_sweep = pd.DataFrame(results)
        
        # Determine optimal threshold (maximizing macro F1 within validation constraints or overall)
        if valid_points:
            valid_df = df_sweep[df_sweep["Meets_Validation"] == True]
            best_row = valid_df.sort_values(by="Macro_F1 (%)", ascending=False).iloc[0]
            status = f"Validation Criteria MET! Valid threshold range: [{min(valid_points):.2f}, {max(valid_points):.2f}]. Best Threshold: {best_row['Threshold']:.2f} (Acc={best_row['Accuracy (%)']:.2f}%, Recall={best_row['Attack_Recall (%)']:.2f}%, FPR={best_row['FPR (%)']:.2f}%)"
        else:
            best_row = df_sweep.sort_values(by="Macro_F1 (%)", ascending=False).iloc[0]
            status = f"Validation Criteria NOT fully met simultaneously. Best Macro-F1 at Threshold: {best_row['Threshold']:.2f}"
            
        print(f"  {status}")
        return df_sweep, best_row, valid_points

