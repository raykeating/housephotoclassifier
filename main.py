import torch
import torchvision
import torchvision.transforms as transforms
import flask
from PIL import Image
from flask_cors import CORS
from pprint import pprint
import os
import json

# Load the model
# model = torchvision.models.resnet18(pretrained=False)
# model.fc = torch.nn.Linear(in_features=512, out_features=6)
# model.load_state_dict(torch.load(r"C:\Users\rayke\Code\photosorter\classifier\model_v0_1.pt"))

# # Move the model to the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

def get_random_word():
    import random
    with open(r"C:\Users\rayke\Code\photosorter\classifier\words.txt", "r") as f:
        words = f.readlines()
    return random.choice(words).strip()


# Create the prediction function
def predict_from_tags(image, classes):

    from clip_client import Client
    from docarray import Document

    # images = [Image.open("C:\\Users\\rayke\\Code\\photosorter\\data\\example\\1.jpg").convert("RGB")]
    # texts = ["Entrance", "Front Yard", "Back Yard", "Exterior of House", "Bathroom", "Bedroom", "Dining Room", "Kitchen", "Living Room", "Other"]

    c = Client('grpc://127.0.0.1:51000')

    # remove alpha channel if it exists
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # create temporary file (this is a hacky way to do this, but it works for now)
    image.save('C:\\Users\\rayke\\Code\\photosorter\\classifier\\images\\temp.jpg')

    d = Document(
        uri='C:\\Users\\rayke\\Code\\photosorter\\classifier\\images\\temp.jpg',
        matches=[
            Document(text=f'{p}')
            for p in classes
        ],
    )

    r = c.rank([d])
    
    # delete temporary file
    os.remove('C:\\Users\\rayke\\Code\\photosorter\\classifier\\images\\temp.jpg')

    return r[0].matches[0].text, r[0].matches[0].scores['clip_score'].value

# Create the Flask app
app = flask.Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})

@app.route("/predict", methods=["POST"])
def predict_api():
    
    # if there is no image in the request, return an error
    if flask.request.files.get("images") is None:
        return flask.jsonify({"error": "No image in the request"}), 400
    
    # if there are no classes in the request, return an error
    if flask.request.form.get("classes") is None:
        return flask.jsonify({"error": "No classes in the request"}), 400

    print("Received request...")
    classes = flask.request.form['classes']
    classes = json.loads(classes)
    
    predictions = []

    for image in flask.request.files.getlist("images"):
        # Read the image via file.stream
        image = Image.open(image.stream)

        # Make the prediction
        prediction, confidence = predict_from_tags(image, classes)
        predictions.append({'prediction': prediction, 'confidence': confidence})
    
    response = {'predictions': predictions}

    # Convert the prediction to JSON and return it
    return flask.jsonify(response)


if __name__ == "__main__":
    print("Starting the server...")
    app.run(port=5000, debug=True)