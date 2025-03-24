import torch
import cv2
import pickle
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from utils import FaceMLP  # Import your FaceMLP model from utils.py
import os 
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load FaceNet model
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
# Preprocessing transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_label_encoder(label_encoder_path):
    with open(label_encoder_path, "rb") as f:
        return pickle.load(f)

def load_model(model_path, num_classes):
    model = FaceMLP(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.eval().to(device)

def extract_embedding(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = facenet(image_tensor).cpu().numpy().flatten()
    
    return embedding

def predict(image_path, model, label_encoder):
    embedding = extract_embedding(image_path)
    input_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class index
    predicted_class_idx = torch.argmax(output, dim=1).item()

    # Decode the label
    person_id = label_encoder.inverse_transform([predicted_class_idx])[0]

    return person_id

if __name__ == "__main__":
    # Specify paths for .pth and .pkl files
    model_path = "/home/user/sarath/clientapi_comm_testing/Face_Authentication_FL_same200_labels/server_aggregated_weights/aggregated_weights_round_100.pth"
    label_encoder_path = "/home/user/sarath/clientapi_comm_testing/Face_Authentication_FL_same200_labels/label_encoder/label_encoder-CL-1.pkl"
    image_path = "/home/user/sarath/clientapi_comm_testing/Face_Authentication_FL_same200_labels/datasets/CL-4/face_images/person-022/22-14.jpg"  # Replace with the test image path

    # Load label encoder and get num_classes
    label_encoder = load_label_encoder(label_encoder_path)
    num_classes = len(label_encoder.classes_)

    # Load the model
    model = load_model(model_path, num_classes)

    # Predict the person ID
    predicted_id = predict(image_path, model, label_encoder)
    image_name = os.path.basename(image_path)
    #print(f"Predicted Person ID: {predicted_id}")
    print(f"Image File: {image_name} | Predicted Person ID: {predicted_id}")
