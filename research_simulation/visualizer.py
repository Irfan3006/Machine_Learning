import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .config import Config

class Visualizer:
    def __init__(self, config=Config):
        self.config = config
        os.makedirs(self.config.FIGURES_DIR, exist_ok=True)
        # Apply standard clean aesthetic
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
        plt.rcParams['axes.edgecolor'] = '#cccccc'
        plt.rcParams['axes.linewidth'] = 0.8

    def plot_rq1_comparison(self, df_test, df_cv):
        """
        Generates comparison figures for RQ1: Model performance and computational cost.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        
        # Plot 1: Accuracy, Macro-F1, Attack Recall, FPR
        models = df_test["Model"].tolist()
        x = np.arange(len(models))
        width = 0.2
        
        ax1 = axes[0]
        rects1 = ax1.bar(x - 1.5*width, df_test["Accuracy (%)"], width, label="Accuracy (%)", color="#1f77b4")
        rects2 = ax1.bar(x - 0.5*width, df_test["Macro F1 (%)"], width, label="Macro F1 (%)", color="#2ca02c")
        rects3 = ax1.bar(x + 0.5*width, df_test["Attack Recall (%)"], width, label="Attack Recall (%)", color="#ff7f0e")
        rects4 = ax1.bar(x + 1.5*width, df_test["False Positive Rate (%)"], width, label="FPR (%)", color="#d62728")
        
        ax1.set_ylabel("Persentase (%)", fontsize=11, fontweight="bold")
        ax1.set_title("(a) Evaluasi Kinerja Klasifikasi (Held-Out Test)", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=10, fontweight="bold")
        ax1.legend(loc="lower right", framealpha=0.9)
        ax1.set_ylim(0, 110)
        
        for bar in rects1 + rects2 + rects3 + rects4:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=7.5, rotation=0)
            
        # Plot 2: Training Time vs Inference Latency
        ax2 = axes[1]
        width2 = 0.35
        rects_t = ax2.bar(x - width2/2, df_test["Training Time (s)"], width2, label="Waktu Latih / Training Time (s)", color="#9467bd")
        
        ax2_twin = ax2.twinx()
        rects_inf = ax2_twin.bar(x + width2/2, df_test["Inference Latency (ms/sample)"], width2, label="Inference Latency (ms/sampel)", color="#8c564b")
        
        ax2.set_ylabel("Waktu Pelatihan (detik)", fontsize=11, fontweight="bold", color="#9467bd")
        ax2_twin.set_ylabel("Inference Latency (ms/sampel)", fontsize=11, fontweight="bold", color="#8c564b")
        ax2.set_title("(b) Biaya Komputasi & Latensi Inferensi", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, fontsize=10, fontweight="bold")
        
        # Add values on top of bars
        for bar in rects_t:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}s", ha='center', va='bottom', fontsize=8, color="#9467bd", fontweight="bold")
        for bar in rects_inf:
            yval = bar.get_height()
            ax2_twin.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0002, f"{yval:.4f}ms", ha='center', va='bottom', fontsize=8, color="#8c564b", fontweight="bold")
            
        fig.tight_layout()
        out_path = os.path.join(self.config.FIGURES_DIR, "rq1_model_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved RQ1 plot to {out_path}")

    def plot_rq2_feature_selection(self, df_subsets, ranking_df):
        """
        Plots RQ2: Trade-off between feature subset size, Macro-F1 retention, and training time.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
        
        # Subplot 1: Top 15 Most Important Features
        ax1 = axes[0]
        top15 = ranking_df.head(15).iloc[::-1]
        bars = ax1.barh(top15["Feature"], top15["Importance"], color="#3b528b", edgecolor="black", linewidth=0.5)
        ax1.set_xlabel("Feature Importance (Random Forest Gini Impurity)", fontsize=10, fontweight="bold")
        ax1.set_title("(a) Top 15 Fitur Paling Berpengaruh (UNSW-NB15)", fontsize=11, fontweight="bold")
        for bar in bars:
            w = bar.get_width()
            ax1.text(w + 0.002, bar.get_y() + bar.get_height()/2.0, f"{w:.3f}", ha='left', va='center', fontsize=8)
            
        # Subplot 2: Macro-F1 vs Training Time across Feature Subsets
        ax2 = axes[1]
        x = df_subsets["Num_Features"]
        
        color1 = "#1f77b4"
        line1 = ax2.plot(x, df_subsets["Macro_F1 (%)"], marker='o', linewidth=2.5, color=color1, label="Macro F1 (%)")
        ax2.set_xlabel("Jumlah Fitur Terpilih (K)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Macro F1 (%)", fontsize=11, fontweight="bold", color=color1)
        ax2.tick_params(axis='y', labelcolor=color1)
        ax2.set_ylim(80, 92)
        ax2.invert_xaxis()
        
        ax2_twin = ax2.twinx()
        color2 = "#d62728"
        line2 = ax2_twin.plot(x, df_subsets["Training_Time_s"], marker='s', linestyle='--', linewidth=2.5, color=color2, label="Waktu Pelatihan (s)")
        ax2_twin.set_ylabel("Waktu Pelatihan (detik)", fontsize=11, fontweight="bold", color=color2)
        ax2_twin.tick_params(axis='y', labelcolor=color2)
        
        # Annotate points
        for i, row in df_subsets.iterrows():
            ax2.annotate(f"{row['Macro_F1 (%)']:.1f}%", (row['Num_Features'], row['Macro_F1 (%)']),
                         textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8, color=color1, fontweight="bold")
            ax2_twin.annotate(f"{row['Training_Time_s']:.1f}s", (row['Num_Features'], row['Training_Time_s']),
                              textcoords="offset points", xytext=(0, -12), ha='center', fontsize=8, color=color2, fontweight="bold")
                              
        ax2.set_title("(b) Trade-Off Macro-F1 vs Biaya Pelatihan (RQ2)", fontsize=11, fontweight="bold")
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="lower right", framealpha=0.9)
        
        fig.tight_layout()
        out_path = os.path.join(self.config.FIGURES_DIR, "rq2_feature_selection.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved RQ2 plot to {out_path}")

    def plot_rq3_threshold(self, df_sweep, best_row, valid_points):
        """
        Plots RQ3: Threshold sweep curves, operating characteristics, and validation criteria fulfillment.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)
        
        # Subplot 1: Accuracy, Recall, and FPR vs Decision Threshold
        ax1 = axes[0]
        th = df_sweep["Threshold"]
        ax1.plot(th, df_sweep["Accuracy (%)"], label="Accuracy (%)", color="#1f77b4", linewidth=2.5)
        ax1.plot(th, df_sweep["Attack_Recall (%)"], label="Attack Recall (%)", color="#2ca02c", linewidth=2.5)
        ax1.plot(th, df_sweep["FPR (%)"], label="False Positive Rate / FPR (%)", color="#d62728", linewidth=2.5)
        ax1.plot(th, df_sweep["Macro_F1 (%)"], label="Macro F1 (%)", color="#9467bd", linestyle=":", linewidth=2)
        
        # Validation target line at 93%
        ax1.axhline(93.0, color="black", linestyle="--", alpha=0.7, label="Target Validasi (≥ 93%)")
        
        # Highlight valid threshold zone if exists
        if valid_points:
            v_min, v_max = min(valid_points), max(valid_points)
            ax1.axvspan(v_min - 0.025, v_max + 0.025, color="green", alpha=0.15, label=f"Zona Target Validasi [{v_min:.2f} - {v_max:.2f}]")
            ax1.axvline(best_row["Threshold"], color="darkgreen", linestyle="-.", linewidth=2, label=f"Threshold Optimal ({best_row['Threshold']:.2f})")
            
        ax1.set_xlabel("Decision Threshold (θ)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Persentase Kinerja (%)", fontsize=11, fontweight="bold")
        ax1.set_title("(a) Kurva Kinerja vs Threshold Keputusan", fontsize=11, fontweight="bold")
        ax1.legend(loc="center left", bbox_to_anchor=(0.02, 0.45), framealpha=0.9, fontsize=8.5)
        ax1.set_xlim(0.05, 0.95)
        ax1.set_ylim(0, 105)
        
        # Subplot 2: Operating Curve (Attack Recall vs FPR)
        ax2 = axes[1]
        sc = ax2.scatter(df_sweep["FPR (%)"], df_sweep["Attack_Recall (%)"], c=df_sweep["Threshold"], cmap="viridis", s=70, edgecolors='black', zorder=4)
        cbar = plt.colorbar(sc, ax=ax2)
        cbar.set_label("Threshold (θ)", fontsize=10, fontweight="bold")
        
        ax2.plot(df_sweep["FPR (%)"], df_sweep["Attack_Recall (%)"], color="gray", linestyle="-", alpha=0.5, zorder=3)
        
        # Highlight default (0.50) vs optimal
        def_row = df_sweep[df_sweep["Threshold"] == 0.50].iloc[0]
        ax2.scatter(def_row["FPR (%)"], def_row["Attack_Recall (%)"], color="red", s=140, marker="X", zorder=5, label=f"Default (θ=0.50): FPR={def_row['FPR (%)']:.1f}%, Rec={def_row['Attack_Recall (%)']:.1f}%")
        
        if valid_points:
            ax2.scatter(best_row["FPR (%)"], best_row["Attack_Recall (%)"], color="lime", s=180, marker="*", edgecolors='black', zorder=6,
                        label=f"Optimal (θ={best_row['Threshold']:.2f}): FPR={best_row['FPR (%)']:.1f}%, Rec={best_row['Attack_Recall (%)']:.1f}%")
            
        # Target zone: Recall >= 93% and Accuracy >= 93% (FPR ~ <= 8%)
        ax2.axhline(93.0, color="darkgreen", linestyle="--", alpha=0.6, label="Recall ≥ 93%")
        ax2.axvline(10.0, color="crimson", linestyle=":", alpha=0.6, label="FPR Target ≤ 10%")
        
        ax2.set_xlabel("False Positive Rate / FPR (%) [False Alarm]", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Attack Recall (%) [Tingkat Deteksi Serangan]", fontsize=11, fontweight="bold")
        ax2.set_title("(b) Operating Curve: Recall Serangan vs False Alarm", fontsize=11, fontweight="bold")
        ax2.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
        ax2.set_xlim(-2, 45)
        ax2.set_ylim(80, 102)
        
        fig.tight_layout()
        out_path = os.path.join(self.config.FIGURES_DIR, "rq3_threshold_tradeoff.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved RQ3 plot to {out_path}")

