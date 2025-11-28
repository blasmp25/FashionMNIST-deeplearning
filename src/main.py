import argparse
import logging
import os
import sys
import subprocess
from datetime import datetime
import shutil

# ===============================
# LOGGING CONFIGURATION
# ===============================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")
log_path = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(console)

logging.info("===== Pipeline started =====")


# ===============================
# CLEAN FUNCTION
# ===============================
def clean_directories():
    """Delete results directories to start fresh."""
    dirs_to_clean = ["models", "plots", "analysis", "experiments"]

    logging.info("→ Cleaning project directories...")

    for d in dirs_to_clean:
        try:
            if os.path.exists(d):
                shutil.rmtree(d)
                logging.info(f"Deleted: {d}")
        except Exception as e:
            logging.error(f"Error deleting {d}: {e}")

        os.makedirs(d, exist_ok=True)
        logging.info(f"Recreated: {d}")

    logging.info("✓ Clean completed.")


# ===============================
# HELPER TO RUN SUBSCRIPTS
# ===============================
def run_script(script_name, description):
    """Run a Python script located inside src/."""
    logging.info(f"→ Starting: {description}...")
    try:
        subprocess.run([sys.executable, f"src/{script_name}"], check=True)
        logging.info(f"✓ Completed: {description}")
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ Error while running {script_name}: {e}")
        sys.exit(1)


# ===============================
# PIPELINE ACTIONS
# ===============================
def run_train():
    run_script("train.py", "Training standard model")

def run_grid():
    run_script("grid_search.py", "Executing grid search experiments")

def run_plots():
    run_script("plot_results.py", "Generating plots")

def run_analysis():
    run_script("analyze_results.py", "Analyzing results")

def run_viz():
    run_script("visualize_predictions.py", "Visualization of predictions")

def run_all():
    logging.info("Executing full pipeline: clean → train → grid → plots → analyze")
    run_train()
    run_grid()
    run_plots()
    run_analysis()
    run_viz()
    logging.info("Full pipeline completed successfully.")


# ===============================
# ARGUMENT PARSER
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Master ML Pipeline (Training + Grid Search + Plots + Analysis + Cleaning)"
    )

    parser.add_argument("--clean", action="store_true", help="Clean models, plots, experiments and analysis folders")
    parser.add_argument("--train", action="store_true", help="Run standard training")
    parser.add_argument("--grid", action="store_true", help="Run grid search")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    parser.add_argument("--analyze", action="store_true", help="Analyze grid search results")
    parser.add_argument("--viz", action="store_true", help="Generate prediction visualizations")
    parser.add_argument("--all", action="store_true", help="Run the full pipeline")

    return parser.parse_args()


# ===============================
# MAIN ENTRY
# ===============================
if __name__ == "__main__":
    args = parse_args()

    # Clean first if requested
    if args.clean:
        clean_directories()

    # Run full pipeline
    if args.all:
        run_all()

    # Individual actions
    if args.train:
        run_train()
    if args.grid:
        run_grid()
    if args.plots:
        run_plots()
    if args.analyze:
        run_analysis()
    if args.viz:
        run_viz()

    # No flags
    if not any(vars(args).values()):
        print("No flags provided. Use --help to see available options.")
        logging.info("No action selected.")

    logging.info("===== Pipeline finished =====")
