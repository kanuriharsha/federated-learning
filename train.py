import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ImageClassifier  # Import model

# Load Dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="data", download=True, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Model
model = ImageClassifier().to(device)
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# Initialize label counts
label_counts = {i: 0 for i in range(10)}  # Since MNIST has 10 classes (0-9)

# Train Model
for epoch in range(5):  # Train for 5 epochs
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update label counts
        for label in labels:
            label_counts[label.item()] += 1

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save Model and Label Counts
torch.save({
    "model_state": model.state_dict(),
    "label_counts": label_counts
}, 'client_model.pt')

print("[*] Training complete. Model and label counts saved as 'client_model.pt'.")
