# client.py
import socket
import torch
import pickle
from model import ImageClassifier

HOST = "10.123.33.52"
PORT = 5001

# Load model and label_counts
checkpoint = torch.load("client_model.pt", map_location=torch.device("cpu"))
client_model = ImageClassifier()
client_model.load_state_dict(checkpoint["model_state"])
label_counts = checkpoint["label_counts"]

# Prepare payload
payload = {
    "state_dict": client_model.state_dict(),
    "label_counts": label_counts
}

# Serialize and send
model_data = pickle.dumps(payload)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((HOST, PORT))
    print(f"[*] Connected to server {HOST}:{PORT}")
    client_socket.sendall(model_data)
    client_socket.shutdown(socket.SHUT_WR)
    print("[*] Model and label counts sent.")
