import os
import json
import torch
from itertools import product

from dataset import create_dataloaders
from engine import train
from tinyvgg import TinyVGG
from utils import save_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define hyperparameters to test
hidden_units_list = [16,32,64]
learning_rates = [0.001, 0.0005]
epochs_list = [5,10]

BATCH_SIZE = 32

def run_experiment(hidden_units, lr, epochs):
    """Trains a model with a specific combination"""
    
    # Create dataloaders
    train_dl, test_dl, class_names = create_dataloaders(
        batch_size=BATCH_SIZE
    )
    
    # Create model
    model = TinyVGG(
        input_shape=1,
        hidden_units=hidden_units,
        output_shape=len(class_names)
    ).to(device)
    
    # Optimizer and loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train
    results = train(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device
    )
    
    # Save the model
    model_name = f"tinyvgg_h{hidden_units}_lr{lr}_e{epochs}.pth"
    save_model(model, target_dir="models", model_name=model_name)
    
    return results, model_name

def main():
    os.makedirs("experiments", exist_ok=True)
    
    experiment_counter = 0
    all_results = []
    
    for hidden, lr, epochs in product(hidden_units_list, learning_rates, epochs_list):
        
        print(f"\n==== EXPERIMENT {experiment_counter} ====")
        print(f"Hidden Units: {hidden}, LR: {lr}, Epochs: {epochs}")

        results, model_file = run_experiment(hidden, lr, epochs)

        experiment_data = {
            "experiment_id": experiment_counter,
            "hidden_units": hidden,
            "learning_rate": lr,
            "epochs": epochs,
            "model_file": model_file,
            "results": results
        }

        all_results.append(experiment_data)
        experiment_counter += 1

    # Save results in a JSON
    with open("experiments/summary.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nGrid search completed. Results saved in experiments/summary.json")
    
    
if __name__ == "__main__":
    main()