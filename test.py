import torch
import torchvision
import matplotlib.pyplot as plt


# Load the model
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(in_features=512, out_features=6)
model.load_state_dict(torch.load("model_v0_1.pt"))

# Move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load the test dataset
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = torchvision.datasets.ImageFolder("./test_dataset", transform=data_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Get the class names
class_names = dataset.classes
print(class_names)

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("Accuracy: %.2f %%" % (100 * correct / total))

import matplotlib.widgets as widgets

# Initialize the figure and axes
fig, ax = plt.subplots()

# Create the next and previous buttons
next_button = widgets.Button(plt.axes([0.81, 0.05, 0.1, 0.075]), 'Next')
prev_button = widgets.Button(plt.axes([0.7, 0.05, 0.1, 0.075]), 'Previous')

# Initialize the index and data iterator
index = 0
dataiter = iter(dataloader)

def next_image(event):
    global index, dataiter
    index += 1
    try:
        images, labels = next(dataiter)
    except StopIteration:
        dataiter = iter(dataloader)
        index = 0
        images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    ax.clear()
    ax.imshow(images[0].cpu().permute(1, 2, 0))
    ax.set_title("Predicted: %s Actual: %s" % (class_names[predicted[0]], class_names[labels[0]]))
    fig.canvas.draw()

def prev_image(event):
    global index, dataiter
    index -= 1
    if index < 0:
        index = len(dataloader) - 1
    dataiter = iter(dataloader)
    for _ in range(index):
        images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    ax.clear()
    ax.imshow(images[0].cpu().permute(1, 2, 0))
    ax.set_title("Predicted: %s Actual: %s" % (class_names[predicted[0]], class_names[labels[0]]))
    fig.canvas.draw()

# Attach the event handlers to the buttons
next_button.on_clicked(next_image)
prev_button.on_clicked(prev_image)

# Show the first image
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, predicted = torch.max(outputs.data, 1)
ax.imshow(images[0].cpu().permute(1, 2, 0))
ax.set_title("Predicted: %s Actual: %s" % (class_names[predicted[0]], class_names[labels[0]]))
plt.show()


