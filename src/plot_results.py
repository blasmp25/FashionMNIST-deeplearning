import json
import os
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
RESULTS_PATH = "experiments/summary.json"   
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_metric(train_values, test_values, metric_name, output_path, title):
    """Plot train and test curves for a metric."""
    plt.figure(figsize=(8, 5))
    epochs = list(range(1, len(train_values) + 1))

    plt.plot(epochs, train_values, label="Train " + metric_name.capitalize(), linewidth=2)
    plt.plot(epochs, test_values, label="Test " + metric_name.capitalize(), linewidth=2)

    plt.xlabel("Epochs")
    plt.ylabel(metric_name.capitalize())
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path, dpi=150)
    plt.close()


def main():

    # =========================
    # LOAD JSON RESULTS
    # =========================
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    
    if isinstance(data, dict):
        print("Single experiment detected. Generating 2 plots...")
        train_loss = data["train_loss"]
        train_acc = data["train_acc"]
        test_loss = data["test_loss"]
        test_acc = data["test_acc"]

        plot_metric(
            train_loss, test_loss,
            "loss",
            os.path.join(OUTPUT_DIR, "loss_curve_single.png"),
            "Loss Curve (Single Experiment)"
        )

        plot_metric(
            train_acc, test_acc,
            "accuracy",
            os.path.join(OUTPUT_DIR, "accuracy_curve_single.png"),
            "Accuracy Curve (Single Experiment)"
        )

        print("Plots saved in:", OUTPUT_DIR)
        return

    # =========================
    # GRID SEARCH CASE (MULTIPLE EXPERIMENTS)
    # =========================

    print(f"Detected {len(data)} experiments.")
    print("Generating plots for each model...")

    for exp in data:
        exp_id = exp["experiment_id"]
        hidden = exp["hidden_units"]
        lr = exp["learning_rate"]
        epochs = exp["epochs"]
        results = exp["results"]

        train_loss = results["train_loss"]
        train_acc = results["train_acc"]
        test_loss = results["test_loss"]
        test_acc = results["test_acc"]

        # Titles for readability
        title_loss = f"Loss - Exp {exp_id} (H={hidden}, LR={lr}, Epochs={epochs})"
        title_acc =  f"Accuracy - Exp {exp_id} (H={hidden}, LR={lr}, Epochs={epochs})"

        # Output file paths
        loss_file = os.path.join(OUTPUT_DIR, f"loss_curve_exp{exp_id}.png")
        acc_file  = os.path.join(OUTPUT_DIR, f"accuracy_curve_exp{exp_id}.png")

        # Generate plots
        plot_metric(train_loss, test_loss, "loss", loss_file, title_loss)
        plot_metric(train_acc, test_acc, "accuracy", acc_file, title_acc)

    print(f"\nAll plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
