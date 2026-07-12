"""
Centralized configuration for the Predictive Maintenance pipeline.
Uses dataclass for type-safe config management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")

    # Azure PdM file names
    telemetry_file: str = "PdM_telemetry.csv"
    errors_file: str = "PdM_errors.csv"
    maint_file: str = "PdM_maint.csv"
    failures_file: str = "PdM_failures.csv"
    machines_file: str = "PdM_machines.csv"

    # Sensor columns
    sensor_cols: List[str] = field(default_factory=lambda: ["volt", "rotate", "pressure", "vibration"])
    id_col: str = "machineID"
    timestamp_col: str = "datetime"

    # Column name mapping (original -> standardized)
    column_mapping: Dict[str, str] = field(default_factory=lambda: {
        "machineID": "machine_id",
        "volt": "voltage",
        "rotate": "rotation_speed",
        "pressure": "pressure",
        "vibration": "vibration",
        "datetime": "timestamp",
        "errorID": "error_id",
        "comp": "component",
        "model": "machine_model",
        "age": "machine_age_years",
        "failure": "failure_type",
    })


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # Rolling window sizes (hours)
    window_sizes: List[int] = field(default_factory=lambda: [6, 12, 24, 48, 72, 168])

    # Lag features (hours behind)
    lag_periods: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 24, 48, 72, 168])

    # Prediction horizon (hours ahead to predict failure)
    prediction_horizon_hours: int = 24

    # Lead time for labeling (hours before failure = positive class)
    failure_lead_time_hours: int = 48

    # Minimum cycles before failure to consider for training
    min_cycles_before_failure: int = 24

    # Feature groups
    statistical_features: List[str] = field(default_factory=lambda: [
        "mean", "std", "min", "max", "skew", "kurtosis",
        "q25", "q50", "q75", "range", "iqr"
    ])

    trend_features: List[str] = field(default_factory=lambda: [
        "slope", "intercept", "r_squared"
    ])

    frequency_features: List[str] = field(default_factory=lambda: [
        "dominant_freq", "spectral_energy", "spectral_entropy"
    ])


@dataclass
class ModelConfig:
    """Model training configuration."""
    # XGBoost
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 12,  # Class imbalance ratio
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "early_stopping_rounds": 50,
        "random_state": 42,
    })

    # Random Forest
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    })

    # LSTM
    lstm_params: Dict[str, Any] = field(default_factory=lambda: {
        "sequence_length": 48,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": True,
        "attention_heads": 4,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "epochs": 100,
        "patience": 15,
    })

    # Training
    test_size: float = 0.2
    validation_size: float = 0.1
    cv_folds: int = 5
    random_seed: int = 42

    # Class imbalance handling
    use_smote: bool = False
    use_class_weights: bool = True
    pos_weight_factor: float = 12.0


@dataclass
class EvaluationConfig:
    """Evaluation & business metrics configuration."""
    # Asymmetric cost matrix (USD)
    cost_true_positive: float = 500.0    # Correctly predicted failure → planned maintenance
    cost_false_positive: float = 500.0   # False alarm → unnecessary inspection
    cost_false_negative: float = 12000.0 # Missed failure → emergency repair + downtime
    cost_true_negative: float = 0.0      # Correctly predicted healthy → no action

    # Downtime cost per hour per machine
    downtime_cost_per_hour: float = 8000.0

    # Annual maintenance budget baseline
    annual_maintenance_budget: float = 2400000.0

    # Target metrics
    target_precision: float = 0.85
    target_recall: float = 0.85
    target_false_alarm_rate: float = 0.05  # Per machine per month


@dataclass
class DashboardConfig:
    """Streamlit dashboard configuration."""
    title: str = "Predictive Maintenance — Fleet Health Monitor"
    port: int = 8501
    refresh_interval_seconds: int = 30

    # Risk thresholds for color coding
    risk_low_threshold: float = 0.2
    risk_medium_threshold: float = 0.5
    risk_high_threshold: float = 0.7

    # KPI targets
    kpi_targets: Dict[str, float] = field(default_factory=lambda: {
        "availability": 0.95,
        "mtbf_hours": 1500,
        "oee": 0.85,
        "schedule_compliance": 0.90,
        "false_alarm_rate": 0.05,
    })


@dataclass
class Config:
    """Master configuration aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    def save(self, path: str = "configs/config.yaml") -> None:
        """Save config to YAML file."""
        import dataclasses
        config_dict = dataclasses.asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str = "configs/config.yaml") -> "Config":
        """Load config from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


# Global config instance
config = Config()
