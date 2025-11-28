import json
import os
import csv
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
RESULTS_PATH = "experiments/summary.json"
OUTPUT_DIR = "analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_experiments():
    """Load the JSON file containing grid search results."""
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)
    return data


def sort_by_accuracy(experiments):
    """Sort experiments by final test accuracy."""
    return sorted(
        experiments,
        key=lambda exp: exp["results"]["test_acc"][-1],
        reverse=True
    )


def save_to_csv(experiments, path):
    """Save a CSV with all experiment info."""
    header = [
        "experiment_id", "hidden_units", "learning_rate", "epochs",
        "final_train_acc", "final_test_acc", "model_file"
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for exp in experiments:
            results = exp["results"]
            writer.writerow([
                exp["experiment_id"],
                exp["hidden_units"],
                exp["learning_rate"],
                exp["epochs"],
                results["train_acc"][-1],
                results["test_acc"][-1],
                exp["model_file"]
            ])


def plot_metric_comparison(experiments, metric_name, filename):
    """Plot test metric for each experiment."""
    plt.figure(figsize=(10, 6))

    exp_ids = [exp["experiment_id"] for exp in experiments]
    final_values = [exp["results"][metric_name][-1] for exp in experiments]

    plt.bar(exp_ids, final_values)

    plt.xlabel("Experiment ID")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"Final Test {metric_name.capitalize()} per Experiment")
    plt.grid(axis="y")

    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def main():
    print("Loading experiments...")
    experiments = load_experiments()

    print(f"{len(experiments)} experiments loaded.")

    print("Sorting experiments by test accuracy...")
    sorted_exps = sort_by_accuracy(experiments)

    # ===============================
    # PRINT TOP 5
    # ===============================
    print("\n===== TOP 5 EXPERIMENTS =====")
    for exp in sorted_exps[:5]:
        print(
            f"Exp {exp['experiment_id']} | "
            f"H={exp['hidden_units']} | LR={exp['learning_rate']} | Epochs={exp['epochs']} | "
            f"Test Acc={exp['results']['test_acc'][-1]:.4f}"
        )

    # ===============================
    # SAVE CSV
    # ===============================
    csv_path = os.path.join(OUTPUT_DIR, "experiments_analysis.csv")
    save_to_csv(sorted_exps, csv_path)
    print(f"\nFull experiment table saved to: {csv_path}")

    # ===============================
    # PLOTS
    # ===============================
    plot_metric_comparison(sorted_exps, "test_acc", "accuracy_por_experimento.png")
    plot_metric_comparison(sorted_exps, "test_loss", "loss_por_experimento.png")

    print(f"Plots saved in: {OUTPUT_DIR}")

    # ===============================
    # BEST MODEL DETAILS
    # ===============================
    best = sorted_exps[0]
    print("\n===== BEST MODEL =====")
    print(f"Experiment ID: {best['experiment_id']}")
    print(f"Hidden units: {best['hidden_units']}")
    print(f"Learning rate: {best['learning_rate']}")
    print(f"Epochs: {best['epochs']}")
    print(f"Final Test Accuracy: {best['results']['test_acc'][-1]:.4f}")
    print(f"Model file: {best['model_file']}")


if __name__ == "__main__":
    main()
