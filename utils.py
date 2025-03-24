import sys
import logging
import os 
import numpy as np
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder
import cv2
#from sklearn.model_selection import train_test_split 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import torch.optim as optim

#facenet = InceptionResnetV1(pretrained='vggface2').eval()

def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(levelname)-5.5s] %(message)s',
                                      datefmt='%Y-%b-%d %H:%M')
    logger = logging.getLogger()
    # **Avoid adding multiple handlers** by checking if they exist
    if logger.hasHandlers():
        logger.handlers.clear()  # **Clear existing handlers**

    logger.setLevel(logging.INFO)  #DEBUG)
    # File logger
    file_handler = logging.FileHandler(filename, mode='w')  # default is 'a' to append
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)  #DEBUG)
    logger.addHandler(file_handler)
    # Stdout logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.INFO) #DEBUG)
    logger.addHandler(std_handler)
    return logger

############################################ for folder type images #####################################################################
def get_data_folder(config):
    if config.dataset == 'face_images':
        data_folder = os.path.join(config.data_path, config.dataset)
    # Initialize FaceNet model
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(config.device)   
    # Preprocessing transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def extract_embedding(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image).unsqueeze(0).to(config.device)
        
        with torch.no_grad():
            embedding = facenet(image_tensor).cpu().numpy().flatten()
        
        return embedding
    
    # Load dataset
    embeddings, labels = [], []
    person_ids = sorted(os.listdir(data_folder))

    for person_id in person_ids:
        person_path = os.path.join(data_folder, person_id)
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            embedding = extract_embedding(img_path)
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(person_id)

    # Convert to NumPy arrays
    embeddings = np.array(embeddings)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Use the entire dataset for training
    X_train = embeddings
    y_train = labels
    
    #X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    return X_train, y_train, label_encoder            #X_train, X_test, y_train, y_test, label_encoder

def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_test_tensor).float().mean().item()
    return accuracy

def extract_numpy_weights(model):
# extract weights from the model in numpy format 
    tensor_weights = model.state_dict()
    numpy_weights = {}

    for k in tensor_weights.keys():
        numpy_weights[k] = tensor_weights[k].detach().cpu().numpy()

    return numpy_weights
  
def convert_np_weights_to_tensor(weights):
    """
    Convert weights from NumPy arrays or PyTorch tensors to PyTorch tensors.
    """
    for k in weights.keys():
        if isinstance(weights[k], np.ndarray):
            weights[k] = torch.from_numpy(weights[k])
        elif isinstance(weights[k], torch.Tensor):
            pass  # Already a torch.Tensor, no need to convert
        else:
            raise TypeError(f"Unsupported type for weights[{k}]: {type(weights[k])}. Must be np.ndarray or torch.Tensor.")
    return weights   

class FaceMLP(nn.Module):
    def __init__(self, num_classes):
        super(FaceMLP, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        #x = self.fc3(x)  # Remove softmax here
        x = self.softmax(self.fc3(x))     
        return x
    
def train_model(model, train_data, logger, config):   #test_data
    X_train, y_train =  train_data  
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(config.device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(config.device)

    #X_test, y_test =  test_data
    #X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(config.device)
    #y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(config.device)


    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Training loop
    for epoch in range(config.n_epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    #if (epoch+1) % 10 == 0:
        #print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    #test_accuracy = utils.evaluate_model(model, X_test_tensor, y_test_tensor)
    if config.verbose:
            print()
            logger.info(f"Trained for {config.n_epochs} epochs, Loss: {loss.item():.4f}")#, Test Accuracy: {test_accuracy:.2f}")

    send_model_weights = extract_numpy_weights(model)

    return send_model_weights
