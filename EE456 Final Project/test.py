# Import necessary libraries
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from vit_pytorch import ViT
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from hyperparameters import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import label_binarize

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the test dataset
test_data = ImageFolder('Fast Food Logo Dataset/logos3/test', transform=transform)

# Create data loader
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the Vision Transformer model
model = ViT(
    image_size = 224,
    patch_size = 32,
    num_classes = 6,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# Load the trained model
model.load_state_dict(torch.load('model.pth'))

# Model evaluation
model.eval()
y_true = []
y_pred = []
y_score = []
test_loss = 0
criterion = torch.nn.CrossEntropyLoss()  #Using CrossEntropyLoss

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())
        y_score.extend(outputs.numpy())

# Calculate average losses
test_loss = test_loss / len(test_loader.dataset)

# Calculate accuracy
test_accuracy = accuracy_score(y_true, y_pred)

# Load training metrics
train_metrics = pd.read_csv('train_metrics.csv')  #Saved metrics

# Plot training and testing accuracy
plt.figure(figsize=(12, 6))
plt.plot(train_metrics['epoch'], train_metrics['train_accuracy'], label='Training Accuracy')
plt.plot([0, train_metrics['epoch'].iloc[-1]], [test_accuracy, test_accuracy], label='Test Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.show()

# Compute ROC curve and ROC area for each class
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5])
n_classes = y_true_bin.shape[1]

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic to Multi-Class')
plt.legend(loc="lower right")
plt.show()

# Print confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
