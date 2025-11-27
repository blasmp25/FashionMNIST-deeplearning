import os
import torch
from dataset import create_dataloaders
from engine import train
from tinyvgg import TinyVGG
from utils import save_model

NUM_EPOCHS = 1
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create dataloaders
train_dataloader, test_dataloader, class_names = create_dataloaders(
    batch_size=BATCH_SIZE
)

# Create TinyVGG model
model = TinyVGG(
    input_shape=1, # grayscale images
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Train the model
results = train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device
)

# Save the model
save_model(
    model=model,
    target_dir="models",
    model_name="model.pth"
)

print(results)