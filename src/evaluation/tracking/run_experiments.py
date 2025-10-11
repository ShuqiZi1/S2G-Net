#!/usr/bin/env python3
"""
Python-based experiment runner - cross-platform alternative to bash script
"""
import subprocess
import sys
import time
from pathlib import Path
import json
import glob
# import torch
# from src.evaluation.tracking.params import analyze_model_complexity


# Model configurations - using minimal tracking integration
MODEL_CONFIGS = {
    'rnn':"python -m experiments.train_ns_lstm --model rnn --ts_mask --add_flat --class_weights --num_workers 0 --add_diag --task los",
    'bilstm': "python -m experiments.train_ns_lstm --bilstm --ts_mask --add_flat --class_weights --num_workers 0 --add_diag --task los --read_best",
    'transformer': "python -m experiments.train_ns_transformer --model transformer --ts_mask --add_flat --add_diag --task los --read_best",
    'mamba': "python -m experiments.train_mamba_only --model mamba --ts_mask --add_flat --add_diag --task los --read_best",
    
    'gnn_gat': "python -m experiments.train_ns_gnn --ts_mask --add_flat --class_weights --gnn_name gat --add_diag --task los --read_best",
    'gnn_sage': "python -m experiments.train_ns_gnn --ts_mask --add_flat --class_weights --gnn_name sage --add_diag --task los --read_best",
    'gnn_mpnn': "python -m experiments.train_ns_gnn --ts_mask --add_flat --class_weights --gnn_name mpnn --add_diag --task los --read_best",
    'graphgps': "python -m experiments.train_graphgps_only --model graphgps --ts_mask --add_flat --add_diag --class_weights --task los --dynamic_g",
    
    'dynamic_lstmgnn_gcn': "python -m experiments.train_dynamic --bilstm --random_g --ts_mask --add_flat --class_weights --add_diag --gnn_name gcn --task los --read_best",
    'dynamic_lstmgnn_gat': "python -m experiments.train_dynamic --bilstm --random_g --ts_mask --add_flat --class_weights --add_diag --gnn_name gat --task los --read_best",
    'dynamic_lstmgnn_mpnn': "python -m experiments.train_dynamic --bilstm --random_g --ts_mask --add_flat --class_weights --add_diag --gnn_name mpnn --task los --read_best",
    'lstmgnn_mpnn': "python -m experiments.train_ns_lstmgnn --bilstm --ts_mask --add_flat --class_weights --gnn_name mpnn --add_diag --task los --read_best",
    'lstmgnn_sage': "python -m experiments.train_ns_lstmgnn --bilstm --ts_mask --add_flat --class_weights --gnn_name sage --add_diag --task los --read_best",
    'lstmgnn_gat': "python -m experiments.train_ns_lstmgnn --bilstm --ts_mask --add_flat --class_weights --gnn_name gat --add_diag --task los --read_best",
    'xgb': "python -m experiments.train_xgb --model xgboost --ts_mask --add_flat --add_diag --task los",
    
    'mamba_gps': "python -m experiments.train_mamba_gps_enhgraph --model mamba-gps --ts_mask --add_flat --class_weights --add_diag --task los --read_best --with_edge_types"
}

SEEDS = [2020, 2021, 2022]
RESULTS_BASE_DIR = "results"
SUMMARY_DIR = "results/sum_results"

# Organize models into categories
BASELINE_MODELS = {
    'rnn','bilstm', 'transformer', 'mamba', 
    'gnn_gat', 'gnn_sage', 'gnn_mpnn', 'graphgps',
    'dynamic_lstmgnn_gcn', 'dynamic_lstmgnn_gat', 'dynamic_lstmgnn_mpnn',
    'lstmgnn_mpnn', 'lstmgnn_sage', 'lstmgnn_gat', 'xgb',
}

OUR_MODELS = {'mamba_gps'}

def get_model_save_path(model_name, seed):
    """Get the appropriate save path for a model based on its category"""
    if model_name in BASELINE_MODELS:
        return Path(RESULTS_BASE_DIR) / "baselines" / model_name / f"seed_{seed}"
    elif model_name in OUR_MODELS:
        return Path(RESULTS_BASE_DIR) / model_name / f"seed_{seed}"
    else:
        return Path(RESULTS_BASE_DIR) / "other" / model_name / f"seed_{seed}"

def check_experiment_prerequisites():
    """Check if necessary files and directories exist"""
    issues = []

    # Check if we're in the right directory
    if not Path("src").exists():
        issues.append("❌ 'src' directory not found. Make sure you're in the project root directory.")

    # Check if experiments directory exists
    if not Path("experiments").exists() and not Path("src/experiments").exists():
        issues.append("❌ 'experiments' directory not found.")

    # Check if key training scripts exist
    scripts_to_check = [
        "experiments/train_ns_lstm.py",
        "experiments/train_mamba_only.py",
        "experiments/train_ns_gnn.py",
        "experiments/train_mamba_gps_enhgraph.py"
    ]

    for script in scripts_to_check:
        if not Path(script).exists() and not Path(f"src/{script}").exists():
            issues.append(f"❌ Training script not found: {script}")

    return issues

def run_single_experiment(model_name, command, seed, timeout_hours=6):
    """Run a single experiment with detailed error handling"""
    print(f"\n🌱 Running {model_name} with seed {seed}")
    
    experiment_start = time.time()   

    # Use the new organized directory structure
    seed_dir = get_model_save_path(model_name, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Use --log_path instead of --results_dir and add version tag
    version_tag = f"{model_name}_seed_{seed}"

    # Build the command with correct parameters
    full_command = f"{command} --seed {seed} --log_path {seed_dir} --version {version_tag}"

    print(f"▶️  Command: {full_command}")
    print(f"📁 Results will be saved to: {seed_dir}")

    try:
        # Run the experiment
        result = subprocess.run(
            full_command.split(),
            capture_output=True,
            text=True,
            timeout=timeout_hours * 3600,
            cwd=Path.cwd()
        )
        
        experiment_end = time.time()
        wall_train_hours = round((experiment_end - experiment_start) / 3600, 4)

        # Save logs regardless of success/failure
        logs_dir = seed_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        with open(logs_dir / "stdout.log", 'w') as f:
            f.write(result.stdout)
        with open(logs_dir / "stderr.log", 'w') as f:
            f.write(result.stderr)

        if result.returncode == 0:
            print(f"✅ Successfully completed {model_name} with seed {seed}")

            # Organize PyTorch Lightning results
            try:
                # Lightning saves to log_path/default/version_tag/
                lightning_results_dir = seed_dir / "default" / version_tag
                if lightning_results_dir.exists():
                    print(f"📁 Found Lightning results in {lightning_results_dir}")

                    # Copy important files to the seed directory root for easy access
                    import shutil
                    for pattern in ["*results.json", "*.pkl", "hparams.yaml"]:
                        import glob
                        for file_path in glob.glob(str(lightning_results_dir / pattern)):
                            dest_path = seed_dir / Path(file_path).name
                            if not dest_path.exists():
                                shutil.copy2(file_path, dest_path)
                                print(f"  📋 Copied {Path(file_path).name}")

                base_runtime = {}
                for f in seed_dir.rglob("runtime.json"):
                    with open(f) as jf:
                        base_runtime = json.load(jf)
                    break

                runtime_info = {
                    "model": model_name,
                    "seed": seed,
                    "command": full_command,
                    "success": True,
                    "train_hours": base_runtime.get("train_hours", wall_train_hours),
                    "epoch_per_sec": base_runtime.get("epoch_per_sec"),
                    "inference_ms_per_patient": base_runtime.get("inference_ms_per_patient"),
                    "peak_vram_GB": base_runtime.get("peak_vram_GB"),
                    "total_params": base_runtime.get("total_params"),
                    "model_size_mb": base_runtime.get("model_size_mb"),
                    "lightning_results_path": str(lightning_results_dir) if lightning_results_dir.exists() else None,
                    "organized_results_path": str(seed_dir)
                }

#               if runtime_info["total_params"] is None:
#                   ckpts = glob.glob(str(seed_dir / "**/*.ckpt"), recursive=True)
#                   if ckpts:
#                       model = torch.load(ckpts[0], map_location="cpu")["state_dict"]
#                       param_info = analyze_model_complexity(model)
#                       runtime_info.update({
#                           "total_params": param_info["total_params"],
#                           "model_size_mb": param_info["model_size_mb"]
#                       })

                with open(seed_dir / "runtime.json", 'w') as f:
                    json.dump(runtime_info, f, indent=2)

            except Exception as e:
                print(f"⚠️  Could not organize results: {e}")

            return True
        else:
            print(f"❌ Failed {model_name} with seed {seed} (return code: {result.returncode})")
            with open(seed_dir / "runtime.json", "w") as f:
                json.dump({
                    "model": model_name,
                    "seed": seed,
                    "command": full_command,
                    "success": False,
                    "train_hours": wall_train_hours
                }, f, indent=2)

            # Show error details
            if result.stderr:
                error_lines = result.stderr.split('\n')
                print("🔍 Error details:")
                for line in error_lines[-15:]:  # Show last 15 lines
                    if line.strip():
                        print(f"    {line}")

            print(f"📝 Full logs saved to {logs_dir}/")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout for {model_name} with seed {seed} (>{timeout_hours}h)")
        return False
    except FileNotFoundError as e:
        print(f"💥 Command not found for {model_name} with seed {seed}: {e}")
        return False
    except Exception as e:
        print(f"💥 Exception for {model_name} with seed {seed}: {e}")
        return False

def run_model_all_seeds(model_name, command):
    """Run a model with all seeds"""
    print(f"\n🔬 Starting experiments for {model_name}")
    print(f"Command template: {command}")

    success_count = 0

    for seed in SEEDS:
        if run_single_experiment(model_name, command, seed):
            success_count += 1

    success_rate = success_count / len(SEEDS) * 100
    print(f"📊 {model_name} completed: {success_count}/{len(SEEDS)} seeds ({success_rate:.1f}%)")

    return success_count

def main():
    """Main experiment runner"""
    print("🚀 Starting comprehensive baseline comparison experiments")
    print(f"📊 Seeds: {SEEDS}")
    print(f"📁 Working directory: {Path.cwd()}")
    print(f"📁 Baselines will be saved to: {Path.cwd() / RESULTS_BASE_DIR / 'baselines'}")
    print(f"📁 Our models will be saved to: {Path.cwd() / RESULTS_BASE_DIR}")
    print(f"📋 Summary directory: {Path.cwd() / SUMMARY_DIR}")

    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    issues = check_experiment_prerequisites()
    if issues:
        print("❌ Prerequisites check failed:")
        for issue in issues:
            print(f"  {issue}")
        print("\n💡 Please fix these issues before running experiments.")
        return

    print("✅ Prerequisites check passed!")

    # Create base directories
    baselines_path = Path(RESULTS_BASE_DIR) / "baselines"
    our_models_path = Path(RESULTS_BASE_DIR)
    summary_path = Path(SUMMARY_DIR)

    # Create directory structure
    baselines_path.mkdir(parents=True, exist_ok=True)
    summary_path.mkdir(parents=True, exist_ok=True)

    print(f"📁 Directory structure:")
    print(f"  - Baselines: {baselines_path.absolute()}")
    print(f"  - Our models: {our_models_path.absolute()}")
    print(f"  - Summaries: {summary_path.absolute()}")

    start_time = time.time()
    total_success = 0
    total_experiments = 0

    print("\n🔥 Starting experiments...")

    # Run all models
    for i, (model_name, command) in enumerate(MODEL_CONFIGS.items(), 1):
        model_type = "Our Model" if model_name in OUR_MODELS else "Baseline"
        print(f"\n{'='*60}")
        print(f"🔬 [{i}/{len(MODEL_CONFIGS)}] Starting {model_name} ({model_type})")
        print(f"{'='*60}")

        success_count = run_model_all_seeds(model_name, command)
        total_success += success_count
        total_experiments += len(SEEDS)

        # Show progress
        progress = (i / len(MODEL_CONFIGS)) * 100
        overall_success_rate = (total_success / total_experiments) * 100
        print(f"📈 Progress: {progress:.1f}% complete | Success rate: {overall_success_rate:.1f}% ({total_success}/{total_experiments})")

    end_time = time.time()
    total_hours = (end_time - start_time) / 3600

    print(f"\n🏁 All training experiments completed!")
    print(f"⏱️  Total time: {total_hours:.2f} hours")
    print(f"📊 Final success rate: {total_success}/{total_experiments} ({total_success/total_experiments*100:.1f}%)")

    # Show directory structure
    print(f"\n📁 Results organization:")
    print(f"    Baselines: {baselines_path.absolute()}")
    for model in sorted(list(BASELINE_MODELS)):
        model_dir = baselines_path / model
        if model_dir.exists():
            seed_count = len([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('seed_')])
            print(f"      └── {model}: {seed_count}/3 seeds")

    print(f"    Our models: {our_models_path.absolute()}")
    for model in sorted(list(OUR_MODELS)):
        model_dir = our_models_path / model
        if model_dir.exists():
            seed_count = len([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('seed_')])
            print(f"      └── {model}: {seed_count}/3 seeds")

    # Generate summary tables
    print(f"\n📋 Generating summary tables...")

    try:
        # Import and run aggregation
        sys.path.append('.')
        from src.evaluation.tracking.aggregate import save_summary_tables

        # We need to update the aggregation to handle the new directory structure
        save_summary_tables(Path(RESULTS_BASE_DIR), Path(SUMMARY_DIR))
        print("✅ Summary tables generated successfully!")

    except Exception as e:
        print(f"❌ Failed to generate summary tables: {e}")
        print("You can run this manually later with:")
        print(f"python -c \"from src.evaluation.tracking.aggregate import save_summary_tables; save_summary_tables(Path('{RESULTS_BASE_DIR}'), Path('{SUMMARY_DIR}'))\"")

    print(f"\n✨ Experiment pipeline completed!")
    print(f"\n📊 Results summary:")
    print(f"    - Baseline results: {baselines_path.absolute()}/<model_name>/seed_<seed>/")
    print(f"    - Our model results: {our_models_path.absolute()}/mamba_gps/seed_<seed>/")
    print(f"    - Summary table: {Path(SUMMARY_DIR).absolute()}/summary.csv")
    print(f"    - Scatter plot data: {Path(SUMMARY_DIR).absolute()}/scatter_data.csv")

    # Show quick preview if summary exists
    summary_file = Path(SUMMARY_DIR) / "summary.csv"
    if summary_file.exists():
        print(f"\n📈 Quick results preview:")
        try:
            with open(summary_file) as f:
                lines = f.readlines()
                if lines:
                    print("Summary table columns:")
                    columns = lines[0].strip().split(',')
                    for i, col in enumerate(columns, 1):
                        print(f"  {i}. {col}")

                    print(f"\nFirst few rows:")
                    for line in lines[:5]:
                        print(f"  {line.strip()}")
        except Exception as e:
            print(f"Could not preview summary: {e}")
    else:
        print(f"\n❌ Summary table not generated - check for errors above")

    # Show failed experiments summary
    if total_success < total_experiments:
        print(f"\n⚠️  Failed experiments summary:")
        for model_name in MODEL_CONFIGS:
            for seed in SEEDS:
                seed_dir = get_model_save_path(model_name, seed)
                stderr_file = seed_dir / "logs" / "stderr.log"
                if stderr_file.exists() and stderr_file.stat().st_size > 0:
                    print(f"    {model_name}/seed_{seed}: Check {stderr_file}")

        print(f"\n💡 To debug a specific failure:")
        print(f"    cat results/baselines/<model_name>/seed_<seed>/logs/stderr.log")
        print(f"    cat results/mamba_gps/seed_<seed>/logs/stderr.log")

if __name__ == "__main__":
    main()
