import socket
import json

UDP_IP = ""  # Bind to all available interfaces
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"[DEBUG] Listening for UDP packets on port {UDP_PORT}...")
while True:
    data, addr = sock.recvfrom(4096)  # Buffer size can be adjusted if needed
    message = json.loads(data.decode('utf-8'))
    print(f"[DEBUG] Received UDP packet from {addr}")
    for marker in message["markers"]:
        print(f"Marker ID: {marker['id']}")
        print("Corner Coordinates:")
        for corner in marker["corners"]:
            print(f"  {corner}")
