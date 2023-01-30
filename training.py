import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

# check if CUDA is available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# print device

print(device)

# Load the pre-trained model
model = torchvision.models.resnet18(pretrained=True)

# Freeze all layers except the last one
for param in model.parameters():
    param.requiresGrad = False
for param in model.fc.parameters():
    param.requiresGrad = True

# Replace the last layer with a new one for 6 classes
model.fc = nn.Linear(in_features=512, out_features=6)

# Move the model to the device
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

# Load the dataset
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# no longer in this directory
dataset = torchvision.datasets.ImageFolder("./dataset", transform=data_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model
for epoch in range(18):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("Epoch %d loss: %.3f" % (epoch + 1, running_loss / len(dataloader)))

# Save the model
torch.save(model.state_dict(), "model_v0_1.pt")
print("Model saved.")
