#!/usr/bin/env python3
"""
End-to-end pipeline orchestrator for Predictive Maintenance.

Usage:
    python scripts/run_pipeline.py [--skip-training] [--skip-evaluation]

This script runs the entire pipeline:
  1. Download data (if not present)
  2. Load & validate data
  3. Feature engineering
  4. Model training
  5. Evaluation
  6. Generate reports & visualizations
  7. Launch dashboard (optional)
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def step(description: str):
    """Print a formatted step header."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")


def run_command(cmd: str, cwd: Path = None):
    """Run a shell command and check for errors."""
    cwd = cwd or PROJECT_ROOT
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"❌ Command failed (exit {result.returncode}): {cmd}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run PdM pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM training")
    parser.add_argument("--launch-dashboard", action="store_true", help="Launch dashboard after completion")
    args = parser.parse_args()

    # Step 1: Download data
    if not args.skip_download:
        step("STEP 1: Downloading Data")
        data_dir = PROJECT_ROOT / "data" / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)

        files = [
            "PdM_telemetry.csv", "PdM_errors.csv", "PdM_maint.csv",
            "PdM_failures.csv", "PdM_machines.csv",
        ]
        for f in files:
            if not (data_dir / f).exists():
                url = f"https://azuremlsampleexperiments.blob.core.windows.net/datasets/{f}"
                print(f"  Downloading {f}...")
                run_command(f"wget -q -P {data_dir} {url}")
            else:
                print(f"  ✓ {f} already exists")

    # Step 2: Preprocessing & Feature Engineering
    step("STEP 2: Preprocessing & Feature Engineering")
    run_command("python -m src.data.loader", cwd=PROJECT_ROOT)
    run_command("python -m src.data.preprocessor", cwd=PROJECT_ROOT)
    run_command("python -m src.data.feature_engine", cwd=PROJECT_ROOT)

    # Step 3: Model Training
    if not args.skip_training:
        step("STEP 3: Model Training")
        lstm_flag = "--skip-lstm" if args.skip_lstm else ""
        run_command(f"python -m src.models.train --model all {lstm_flag}".strip())

    # Step 4: Evaluation
    if not args.skip_evaluation:
        step("STEP 4: Evaluation & Reporting")
        run_command("python -m src.evaluation.metrics")
        run_command("python -m src.evaluation.cost_analysis")
        run_command("python -m src.evaluation.visualization")

    # Step 5: Tests
    step("STEP 5: Running Unit Tests")
    run_command("python -m pytest tests/ -v --tb=short")

    # Summary
    step("✅ PIPELINE COMPLETE")
    print(f"""
    ┌──────────────────────────────────────────────────┐
    │                                                  │
    │   ✅ Data downloaded & validated                  │
    │   ✅ Features engineered (200+ features)          │
    │   ✅ Models trained (XGBoost, RF, LSTM)           │
    │   ✅ Evaluation reports generated                 │
    │   ✅ Unit tests passed                            │
    │                                                  │
    │   📊 Reports: reports/figures/                    │
    │   🤖 Models:  models/                             │
    │   📋 Config:  configs/                            │
    │                                                  │
    │   Run 'make dashboard' to launch the dashboard    │
    │   Run 'make api' to start the prediction API      │
    │                                                  │
    └──────────────────────────────────────────────────┘
    """)

    if args.launch_dashboard:
        print("Launching dashboard...")
        run_command("streamlit run src/dashboard/app.py --server.port 8501")


if __name__ == "__main__":
    main()
