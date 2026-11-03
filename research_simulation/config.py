import os

class Config:
    # Random Seed for reproducibility
    RANDOM_SEED = 42
    
    # Dataset Paths
    TRAIN_PATH = "UNSW_NB15_training-set.csv"
    TEST_PATH = "UNSW_NB15_testing-set.csv"
    
    # Evaluation settings
    N_FOLDS = 5
    
    # Column definitions
    TARGET_COL = "label"
    DROP_COLS = ["id", "attack_cat", "label"]
    CATEGORICAL_COLS = ["proto", "service", "state"]
    
    # Output directories
    RESULTS_DIR = "results"
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
    TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
    
    # Validation constraints
    VALIDATION_TARGET_ACCURACY = 0.93
    VALIDATION_TARGET_RECALL = 0.93
    
    # Feature subsets for RQ2
    FEATURE_SUBSETS = [42, 30, 20, 15, 10, 5]
    
    # Threshold sweep grid for RQ3
    THRESHOLDS = [round(x, 2) for x in [i * 0.05 for i in range(1, 20)]] # 0.05 to 0.95

