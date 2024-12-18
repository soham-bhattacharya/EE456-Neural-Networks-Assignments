# Import necessary libraries*
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from vit_pytorch import ViT
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hyperparameters import *

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, +*224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotat/ion(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load the dataset
data = ImageFolder('Fast Food Logo Dataset/logos3', transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_data, val_data = random_split(data, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Initialize the Vision Transformer model
model = ViT(
    image_size = 224,
    patch_size = 32,
    num_classes = 6,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.,
    emb_dropout = 0.1
)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Define learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Prepare lists for saving metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop with early stopping
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):  # Number of epochs
    train_loss = 0.0
    val_loss = 0.0
    train_correct = 0
    val_correct = 0

    # Training
    model.train()
    for images, labels in tqdm(train_loader, desc="Training"):
        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step(val_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_accuracy = 100 * train_correct / len(train_data)
    val_accuracy = 100 * val_correct / len(val_data)

    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}, Training Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}')

    # Save metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model parameters
        torch.save(model.state_dict(), 'model.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print('Early stopping')
        break

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

# Save training metrics
metrics = pd.DataFrame({
    'epoch': np.arange(len(train_losses)) + 1,
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_accuracy': train_accuracies,
    'val_accuracy': val_accuracies
})
metrics.to_csv('train_metrics.csv', index=False)

# Plot training and validation loss over epochs
plt.figure(figsize=(12, 6))
plt.plot(metrics['epoch'], metrics['train_loss'], label='Training Loss')
plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

# Plot training and validation accuracy over epochs
plt.figure(figsize=(12, 6))
plt.plot(metrics['epoch'], metrics['train_accuracy'], label='Training Accuracy')
plt.plot(metrics['epoch'], metrics['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.show()



