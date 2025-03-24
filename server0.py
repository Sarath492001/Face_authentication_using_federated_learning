from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import requests
import threading
import os
import time
import json
# Clear any existing proxy settings in the environment
[os.environ.pop(proxy, None) for proxy in ['HTTP_PROXY', 'HTTPS_PROXY', 'SOCKS_PROXY', 'http_proxy', 'https_proxy', 'socks_proxy', 'all_proxy', 'ALL_PROXY']]

# Initialize FastAPI application
app = FastAPI()

# Event to signal shutdown
shutdown_event = threading.Event()
# Load training configuration
try:
    with open("training_config.json", "r") as file:
        training_config = json.load(file)  # Load JSON as dictionary

        # Extract port and total rounds
        port = training_config.get("port", 8005)  # Default to 8002 if not found
        total_rounds = training_config.get("n_rounds", 100)  # Default to 100 if not found

        print("Loaded training_config.json successfully.")
except FileNotFoundError:
    print("Error: training_config.json not found.")

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
    """
    Endpoint to receive model weights as a .pth file and associated metadata.
    """
    # Use Pydantic for structured metadata if you need additional validation
    message = Message(sender_ip=sender_ip, sender_id=sender_id, round_number=round_number)

    # Directory to save received model weights
    received_weights_dir = "received_model_weights_from_all_clients"
    os.makedirs(received_weights_dir, exist_ok=True)

    # Save the model weights file
    file_path = os.path.join(received_weights_dir, f"received_model_weights_round_{message.round_number}_client_{message.sender_id}.pth")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    print(f"Client {message.sender_id} ({message.sender_ip}) sent model weights for round {message.round_number}")
    # Lock to safely append received messages
    with received_messages_lock:
        received_messages.append({"metadata": message, "file_path": file_path})

    return {"status": "Model weights received successfully"}

'''
################################### Euclidean Distance from the Global Model ##############################################
def load_client_weights(file_path):
    """Load and flatten model weights from a .pth file."""
    state_dict = torch.load(file_path, map_location="cpu")
    # Convert NumPy arrays to PyTorch tensors and flatten all layers
    weight_tensors = [torch.tensor(param).flatten() if isinstance(param, np.ndarray) else param.flatten()
                      for param in state_dict.values()]
    weights = torch.cat(weight_tensors)
    return weights

def compute_global_model(all_model_weights):
    """Compute the global model by averaging all client weights."""
    num_clients = len(all_model_weights)
    global_model = sum(all_model_weights) / num_clients
    return global_model

def compute_euclidean_distance(vec1, vec2):
    """Compute Euclidean distance between two vectors."""
    return torch.norm(vec1 - vec2).item()

def identify_potential_poisoning_ed(all_model_weights):
    """Identify potential poisoned clients using Euclidean distance from the global model."""
    num_clients = len(all_model_weights)

    # Convert model weight dictionaries to flattened tensors
    weight_vectors = [torch.cat([param.flatten() for param in weights.values()]) for weights in all_model_weights]

    # Compute global model by averaging all client weights
    global_model = compute_global_model(weight_vectors)

    # Compute Euclidean distance from each client to the global model
    distances = [compute_euclidean_distance(weights, global_model) for weights in weight_vectors]

    # Print all Euclidean distances
    for idx, distance in enumerate(distances):
        print(f"Client {idx + 1}: Euclidean Distance = {distance:.4f}")

    # Identify the most anomalous client (largest distance)
    poisoned_client = np.argmax(distances)

    print(f"🚨 Potential poisoned client: {poisoned_client + 1} (Max distance: {distances[poisoned_client]:.4f})")
    dissimilar_clients = [poisoned_client]  # Exclude poisoned clients
    return dissimilar_clients
 
################################## Euclidean Distance from the Global Model aggregator function ##############################################

def aggregate_model_weights():
    """Aggregate model weights from clients, excluding detected poisoned clients."""
    all_model_weights = []

    # Sort received messages based on sender_id to ensure consistent ordering
    with received_messages_lock:
        received_messages.sort(key=lambda x: x["metadata"].sender_id)

    for message in received_messages:
        weights = torch.load(message["file_path"], map_location=torch.device("cpu"))
        for key in weights:
            weights[key] = torch.tensor(weights[key]) if isinstance(weights[key], np.ndarray) else weights[key]
        all_model_weights.append(weights)

    # Identify poisoned clients using Euclidean distance method
    dissimilar_clients = identify_potential_poisoning_ed(all_model_weights)

    # Get valid client indices (excluding poisoned clients)
    valid_clients = [i for i in range(len(all_model_weights)) if i not in dissimilar_clients]

    if not valid_clients:
        print("⚠️ No valid clients remaining after filtering! Using all clients.")
        valid_clients = list(range(len(all_model_weights)))

    print("\n✅ Clients used for aggregation:")
    for i in valid_clients:
        print(f"   - Client {i+1}")

    # Initialize final averaged model weights
    final_avg_weights = {key: torch.zeros_like(all_model_weights[0][key]) for key in all_model_weights[0].keys()}
    
    for i in valid_clients:
        for k in final_avg_weights.keys():
            if not k.endswith("num_batches_tracked"):  # Exclude batch tracking keys
                final_avg_weights[k] += all_model_weights[i][k]

    # Average over valid clients
    for k in final_avg_weights.keys():
        if not k.endswith("num_batches_tracked"):
            final_avg_weights[k] /= len(valid_clients)

    return final_avg_weights 

'''
###################### model aggregate function without poison client detection ############################################
def aggregate_model_weights():
    all_model_weights = []
    for message in received_messages:
        weights = torch.load(message["file_path"], map_location=torch.device("cpu"))
        for key in weights:
            weights[key] = torch.tensor(weights[key]) if isinstance(weights[key], np.ndarray) else weights[key]
        all_model_weights.append(weights)

    new_avg_weights = {key: torch.zeros_like(all_model_weights[0][key]) for key in all_model_weights[0].keys()}
    for i in range(len(all_model_weights)):
        for k in new_avg_weights.keys():
            if not k.endswith("num_batches_tracked"):
                new_avg_weights[k] += all_model_weights[i][k]
    for k in new_avg_weights.keys():
        if not k.endswith("num_batches_tracked"):
            new_avg_weights[k] /= len(clients)
    return new_avg_weights
################################################################################################################################

def broadcast_aggregated_weights(weights, round_number):
    # Create a directory to store aggregated weights if it doesn't exist
    save_dir = "server_aggregated_weights"
    os.makedirs(save_dir, exist_ok=True)

    # Define the path to save the model weights
    model_weights_path = os.path.join(save_dir, f"aggregated_weights_round_{round_number}.pth")
    torch.save(weights, model_weights_path)

    for client_id, client_ip in clients.items():
        url = f"http://{client_ip}:{port}/send_message"
        with open(model_weights_path, "rb") as f:
            files = {"file": f}
            payload = {"sender_ip": server_ip, "sender_id": server_id, "round_number": round_number}
            response = requests.post(url, data=payload, files=files)
            print()
            #print(f"Broadcasted aggregated weights to Client {client_id} ({client_ip}) for round {round_number}: {response.json()}")
    print()        
    print(f"Weights Aggregated for round {round_number} and broadcasted to all clients.")
    print()

def start_receiving_messages(server_id, server_ip):

    start_round = 1
    for round_number in range(start_round, total_rounds + 1):
        print(f"Starting Round {round_number}")
        print()
        while len(received_messages) < len(clients):
            time.sleep(5)
        aggregated_weights = aggregate_model_weights()
        broadcast_aggregated_weights(aggregated_weights, round_number)
        with received_messages_lock:
            received_messages.clear()
    print()       
    print("All rounds completed") 
    # Trigger shutdown after the last round
    shutdown_event.set()


if __name__ == "__main__":
    import uvicorn
    import torch
    import os
    import numpy as np
    import warnings
    # Suppress the specific FutureWarning from torch.load
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=UserWarning)

    my_ip = server_ip
    print()
    print(f"This is Server {server_id} and the IP: {server_ip}")
    print()
    # Initialize configuration

    # Configure device
    #config.device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
 
    # Start the thread for sending messages
    threading.Thread(target=start_receiving_messages, args=(server_id, my_ip), daemon=True).start()
   
    # Start the Uvicorn server to handle incoming requests
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port))
    threading.Thread(target=server.run, daemon=True).start()
    # Wait for the shutdown signal
    shutdown_event.wait()
    print("Stopping Server...")
    server.should_exit = True
