from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import requests
import threading
import time
import os
import json
import argparse
# Clear any existing proxy settings in the environment
[os.environ.pop(proxy, None) for proxy in ['HTTP_PROXY', 'HTTPS_PROXY', 'SOCKS_PROXY', 'http_proxy', 'https_proxy', 'socks_proxy', 'all_proxy', 'ALL_PROXY']]

# Initialize FastAPI application
app = FastAPI()

# Event to signal shutdown
shutdown_event = threading.Event()

# Client configuration model using Pydantic for validation
class ClientConfig(BaseModel):

    # DATA AND RESULT PATH
    data_path: str = "./datasets"    
    result_path: str = "./results" 
          
    # TRAINING PARAMETERS 
    dataset: str = "face_images"               #"Face_images" - face images in npz format | "medical_images_2" - 2 class cancer cell images in folder format | "face_images_2" - face images in folder format
    image_size: tuple = (160, 160)               # target image size should be used for training  (BASED ON THE IMAGE SIZE YOU HAVE TO CHANGE THE MODEL PROCESS THE INPUT IMAGE SIZE)
    lr: float = 0.0001                            # 0.001 - face images | 0.00001 - medical images | MODIFY AS PER YOUR USE CASE 
    n_epochs: int = 1000                            # no of epochs 1 - 2000
    n_rounds: int = 100                            # no of communication rounds 1 - 500   
    batch_size: int = 250                        # keep less for vit and more layers cnn models
    n_class: int = 0                             #  Automatically selected
    device: int = 0                              # cpu or gpu selection ( Automatically selected )
    verbose: int = 1                             # for logging 
    port: int = 8005                             # port number for server & clients to communicate | USE 8000 - 8999  | should be same for clients and server 
    max_retries: int = 10                        # Maximum number of times to retry communication in case of failure
    retry_delay: int = 10                        # Time (in seconds) to wait before retrying after a failure
    
client_id = 3                                    # choose the id of this client 
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="Path to config file")
args = parser.parse_args()    
# Determine which config file to use
config_file = args.config if args.config else "training_config.json"
## Load JSON file with error handling
try:
    with open(config_file, "r") as file:
        config_data = json.load(file)
        print(f"Loaded config from: {config_file}")
except FileNotFoundError:
    print(f"Error: {config_file} not found. Using default configuration within the client script.")
    config_data = {}

# All client machines and server should be in the same network, define all the clients ip for quick id selection
# Load clients config from JSON file
try:
    with open("all_clients_ips.json", "r") as file:
        data = json.load(file)  # Loads as a dictionary
        # Extract server info
        server_id = data["server"]["id"]
        server_ip = data["server"]["ip"]
        # Extract client info and convert keys to integers
        clients = {int(k): v for k, v in data["clients"].items()}  # Convert keys to integers
        print("Loaded all_clients_ips.json successfully.")
except FileNotFoundError:
    print("Error: all_clients_ips.json not found.")

# List to store received messages and a lock for thread safety
received_messages = []
received_messages_lock = threading.Lock()

# Message model for incoming messages
class Message(BaseModel):
    sender_ip: str
    sender_id: int
    round_number: int
    
@app.post("/send_message")
async def receive_message(
    sender_ip: str = Form(...),
    sender_id: int = Form(...),
    round_number: int = Form(...),
    file: UploadFile = File(...)
):

    # Use Pydantic for structured metadata if you need additional validation
    message = Message(sender_ip=sender_ip, sender_id=sender_id, round_number=round_number)
    # Parent directory
    parent_dir = "received_weights_from_server"
    # Directory to save received model weights
    received_weights_dir = os.path.join(parent_dir, f"received_model_weights_from_server_CL-{client_id}")
    os.makedirs(received_weights_dir, exist_ok=True)

    # Save the model weights file
    file_path = os.path.join(received_weights_dir, f"received_model_weights_round_{message.round_number}_server.pth")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    print(f"server {message.sender_id} ({message.sender_ip}) sent model weights for round {message.round_number}")
    # Lock to safely append received messages
    with received_messages_lock:
        received_messages.append({"metadata": message, "file_path": file_path})

    return {"status": "Model weights received successfully"}

def save_model_weights_as_pth(model_weights, filepath):  
    # Parent directory
    parent_dir = "checkpoints"
    checkpoints_dir = os.path.join(parent_dir, f"checkpoints-CL-{client_id}")
    # Create the subdirectory if it doesn't exist
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Update filepath to save in the checkpoints directory
    filepath = os.path.join(checkpoints_dir, filepath)

    # Save the model weights
    torch.save(model_weights, filepath)
    #print("Model weights saved in checkpoints directory")

#################################### Retry mechanism ###################################################################################
def send_message_to_peer(peer_ip, sender_id, sender_ip, round_number, model_weights):

    max_retries = config.max_retries
    retry_delay = config.retry_delay
    model_weights_path = f"model_weights_round_{round_number}_client_{sender_id}.pth"
    save_model_weights_as_pth(model_weights, model_weights_path)

    for attempt in range(1, max_retries + 1):
        try:
            url = f"http://{peer_ip}:{config.port}/send_message"
            payload = {
                "sender_ip": sender_ip,
                "sender_id": sender_id,
                "round_number": round_number  
            }
            # Parent directory
            parent_dir = "checkpoints"
            sub_dir = os.path.join(parent_dir, f"checkpoints-CL-{client_id}")
            # Create the subdirectory path to the model weights
            model_weights_full_path = os.path.join(sub_dir, model_weights_path)

            with open(model_weights_full_path, "rb") as f:
                files = {"file": f}

                # Send the request to the peer client
                response = requests.post(url, data=payload, files=files)
            #response = requests.post(url, json=payload, proxies=proxies)
            if response.status_code == 200:
                json_response = response.json()
                print(f"Model weights sent to server({peer_ip}): {json_response}")
                break  # Exit the retry loop on success
            else:
                raise requests.exceptions.RequestException(f"Received status code {response.status_code}")
        
        except requests.exceptions.RequestException as e:
            peer_id = next((client_id for client_id, ip in clients.items() if ip == peer_ip), None)
            print()
            print(f"Attempt {attempt} failed to send model weights to server({peer_ip}): {e}")
            
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)  # Wait before retrying
            else:
                print(f"Failed to send model weights to server({peer_ip}) after {max_retries} attempts.")
                shutdown_event.set()
#######################################################################################################################################

def wait_for_message(sender_id, round_number):
    """
    Function to wait for a message from a specific sender for a given round.
    It checks the received messages and returns the message when found.
    """
    while True:
        with received_messages_lock:
            for message in received_messages:
                if message["metadata"].sender_id == sender_id and message["metadata"].round_number == round_number:
                    return message
        time.sleep(1)  # Avoid busy-waiting

def start_sending_messages(model, client_id, my_ip, config, train_data):  #test_data):

    start_round = 1

    total_rounds = config.n_rounds

    for round_number in range(start_round, total_rounds + 1):
            
            send_model_weights = utils.train_model(model, train_data, logger, config)   #test_data
        
            # Function to send model weights
            def send_weights():
                recipient_ip = server_ip
                #print(f"Sending model weights to server({recipient_ip})...")
                send_message_to_peer(recipient_ip, client_id, my_ip, round_number, send_model_weights)

            # Function to receive model weights
            def receive_weights():
                sender_id = server_id 
                #print(f"Waiting for model weights from server for round {round_number}...")
                message = wait_for_message(sender_id, round_number)
                #print(f"Received model weights from server for round {round_number}.")
                #print(f"Loading the received model weights into the model for round {round_number}")
                received_weights = torch.load(message["file_path"])
        
                model.load_state_dict(utils.convert_np_weights_to_tensor(received_weights))
                print(f"Model weights successfully loaded from server for round {round_number}")

            print()
            print(f"Starting Round {round_number}")
            time.sleep(5)  # Wait before starting

            # Start threads for sending and receiving
            send_thread = threading.Thread(target=send_weights)
            receive_thread = threading.Thread(target=receive_weights)

            send_thread.start()
            receive_thread.start()

            # Wait for both threads to complete
            send_thread.join()
            receive_thread.join()


            print(f"Round {round_number} completed")
            print()  # Add a blank line after round completion

            # Clear messages for this round
            with received_messages_lock:
                received_messages[:] = [m for m in received_messages if m["metadata"].round_number != round_number]
        
            time.sleep(5)  # Wait before starting the next round

    print()       
    print("All rounds completed") 
    # Signal the end of execution
    print()
    print("Shutting down...")
    shutdown_event.set()

            
if __name__ == "__main__":
    import uvicorn
    import torch
    import os
    import utils
    import numpy as np
    from utils import FaceMLP
    import warnings
    import pickle

    # Suppress the specific FutureWarning from torch.load
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=UserWarning)

    my_ip = clients[client_id]
    print()
    print(f"This is Client {client_id} and the IP: {my_ip}")
    print()
    # Initialize configuration
    config = ClientConfig(**config_data)

    # Configure device
    config.device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    config.data_path = f"./datasets/CL-{client_id}" 
    config.result_path = f"./results/results-CL-{client_id}"
    # Create result directory path
    result_path_components = [
        config.result_path,
        config.dataset,
        #f"batch_size_{config.batch_size}",
        #f"lr_{config.lr}",
        f"n_epochs_{config.n_epochs}",
        f"n_rounds_{config.n_rounds}",
    ]

    result_path = os.path.join(*result_path_components)
    os.makedirs(result_path, exist_ok=True)

    # Data preparation based on the selected dataset
    if config.dataset == "face_images":
        print("Using Face Images folder")
        #X_train, X_test, y_train, y_test, label_encoder = utils.get_data_folder(config) 
        X_train, y_train, label_encoder = utils.get_data_folder(config)
    else:
        raise ValueError("Invalid choice for dataset.")   

    print(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")
    #print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")
    print()
    # Parent directory for label encoders
    parent_dir = "label_encoder"
    # Path to store the specific client's label encoder
    label_encoder_path = os.path.join(parent_dir, f"label_encoder-CL-{client_id}.pkl")
    # Create the directory if it doesn't exist
    os.makedirs(parent_dir, exist_ok=True)
    # Save LabelEncoder for decoding predictions
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print("LabelEncoder pickle file saved successfully!")
    print()
    # Ensure `config.n_class` is correctly set before poisoning
    config.n_class = len(np.unique(y_train))
    print(f"Number of unique persons in the Dataset: {config.n_class}")
    print()
    train_data = (X_train, y_train)             
    #test_data = (X_test, y_test)

    logger = utils.get_logger(os.path.join(result_path, f"client_{client_id}.log"))
    print()
    logger.info(f"Hyperparameter setting = {config}")
    print()
    model = FaceMLP(config.n_class).to(config.device)
    # Start the thread for sending messages
    threading.Thread(target=start_sending_messages, args=(model, client_id, my_ip, config, train_data), daemon=True).start()         #test_data
   
    # Start the Uvicorn server to handle incoming requests
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=config.port))
    threading.Thread(target=server.run, daemon=True).start()

    # Wait for the shutdown signal
    shutdown_event.wait()
    print("Stopping server...")
    server.should_exit = True

