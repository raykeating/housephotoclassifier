import torch
import torchvision
import torchvision.transforms as transforms
import flask
from PIL import Image
from flask_cors import CORS

# Load the model
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(in_features=512, out_features=6)
model.load_state_dict(torch.load("model_v0_1.pt"))

# Move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Create the prediction function
def predict(image):

    # Pre-processing the image
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = img_transform(image).unsqueeze(0).to(device)

    # Make the prediction
    output = model(img)
    probs = torch.nn.functional.softmax(output, dim=1)

    # Return the prediction
    classes = ['bathroom', 'bedroom', 'dining_room', 'exterior', 'kitchen', 'living_room']

    # if the confidence is less than 0.5, return 'unknown'
    prediction = classes[torch.argmax(probs)] if torch.max(probs) > 0.6 else 'other'

    probs = probs.detach().cpu().numpy().tolist()[0]
    probs = {classes[i]: probs[i] for i in range(len(classes))}

    return prediction, probs

# Create the Flask app
app = flask.Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

@app.route("/predict", methods=["POST"])
def predict_api():

    predictions = []

    for image in flask.request.files.getlist("image"):
        # Read the image via file.stream
        image = Image.open(image.stream)

        # Make the prediction
        prediction, probs = predict(image)
        predictions.append({'prediction': prediction, 'probs': probs})
    
    response = {'predictions': predictions}

    # Convert the prediction to JSON and return it
    return flask.jsonify(response)

if __name__ == "__main__":
    print("Starting the server...")
    app.run(port=5000)