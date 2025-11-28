import os
import argparse
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import transforms

from dataset import create_dataloaders
from tinyvgg import TinyVGG

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

def find_best_experiment(summary_path="experiments/summary.json"):
    """Return the experiment dict with the highest final test accuracy."""
    if not os.path.exists(summary_path):
        return None

    with open(summary_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        return None

    best = max(data, key=lambda exp: exp["results"]["test_acc"][-1])
    return best


def load_model_from_experiment(exp, class_names, models_dir="models"):
    """
    Recreate the model with the experiment hyperparams (hidden_units) and load its weights.
    exp: experiment dict with keys 'hidden_units' and 'model_file'
    """
    hidden = exp["hidden_units"]
    model_file = exp["model_file"]
    model_path = os.path.join(models_dir, model_file) if not os.path.isabs(model_file) else model_file

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = TinyVGG(input_shape=1, hidden_units=hidden, output_shape=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, model_path


def load_model_from_path(model_path, class_names, hidden_units_fallback=32):
    """
    Load a model from a given .pth path. You need to set hidden units if model
    architecture is not known; we try to guess (fallback).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Use fallback hidden units (you can change if needed)
    model = TinyVGG(input_shape=1, hidden_units=hidden_units_fallback, output_shape=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, model_path


def show_predictions_grid(model, dataloader, class_names, save_path, num_images=16):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(12, 12))

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)

            for i in range(len(X)):
                if images_shown >= num_images:
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=150)
                    plt.close()
                    return

                plt.subplot(4, 4, images_shown + 1)
                img = X[i].cpu().squeeze()
                # if input was normalized, convert back (assuming ToTensor only)
                plt.imshow(img, cmap="gray")
                pred = preds[i].item()
                true = y[i].item()
                color = "green" if pred == true else "red"
                plt.title(f"P: {class_names[pred]}\nT: {class_names[true]}", color=color, fontsize=9)
                plt.axis("off")
                images_shown += 1

    # If dataset smaller than num_images
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(model, dataloader, class_names, save_path):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main(args):
    os.makedirs("plots", exist_ok=True)

    # create dataloaders (we just need test loader and class names)
    _, test_dataloader, class_names = create_dataloaders(batch_size=args.batch_size)

    model = None
    model_path = None

    # priority: explicit model path > chosen exp id > best exp in summary.json > fallback model.pth
    if args.model_path:
        print(f"Loading model from path: {args.model_path}")
        model, model_path = load_model_from_path(args.model_path, class_names, hidden_units_fallback=args.hidden_units_fallback)

    elif args.exp_id is not None:
        # load specific experiment by id
        summary_path = "experiments/summary.json"
        if not os.path.exists(summary_path):
            raise FileNotFoundError("No summary.json found in experiments to load experiments from.")
        with open(summary_path, "r") as f:
            data = json.load(f)
        exp_match = next((e for e in data if e["experiment_id"] == args.exp_id), None)
        if exp_match is None:
            raise ValueError(f"Experiment id {args.exp_id} not found in summary.json")
        model, model_path = load_model_from_experiment(exp_match, class_names)

    else:
        best = find_best_experiment()
        if best:
            print(f"Best experiment found: id={best['experiment_id']} (hidden={best['hidden_units']}, lr={best['learning_rate']}, epochs={best['epochs']})")
            model, model_path = load_model_from_experiment(best, class_names)
        else:
            # fallback to default model
            fallback = "models/model.pth"
            if os.path.exists(fallback):
                print(f"No experiments summary found. Loading fallback model: {fallback}")
                model, model_path = load_model_from_path(fallback, class_names, hidden_units_fallback=args.hidden_units_fallback)
            else:
                raise FileNotFoundError("No model found to visualize. Please train a model or supply --model-path.")

    print(f"Using model: {model_path}")

    # Generate predictions grid and confusion matrix
    predictions_save = os.path.join("plots", f"predictions_{os.path.basename(model_path).replace('.pth','')}.png")
    confusion_save = os.path.join("plots", f"confusion_{os.path.basename(model_path).replace('.pth','')}.png")

    show_predictions_grid(model, test_dataloader, class_names, predictions_save, num_images=args.num_images)
    print(f"Saved predictions grid to {predictions_save}")

    plot_confusion_matrix(model, test_dataloader, class_names, confusion_save)
    print(f"Saved confusion matrix to {confusion_save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predictions and confusion matrix for the best (or chosen) model.")
    parser.add_argument("--model-path", type=str, default=None, help="Direct path to a .pth model file to visualize.")
    parser.add_argument("--exp-id", type=int, default=None, help="Experiment ID from summary.json to load model/hparams for.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for the dataloader used in visualization.")
    parser.add_argument("--num-images", type=int, default=16, help="Number of images to show in the predictions grid (default 16).")
    parser.add_argument("--hidden-units-fallback", type=int, default=32, help="Hidden units if loading model directly from path without hparams info.")
    args = parser.parse_args()
    main(args)
