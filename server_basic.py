import socket
import time

# Define the host and port
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 12345  # Your chosen port number

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the address and port
server_socket.bind((HOST, PORT))

# Enable the server to accept connections
server_socket.listen(1)  # Allow one connection at a time

print(f"Server listening on {HOST}:{PORT}")

# Accept a connection from a client
conn, addr = server_socket.accept()
print(f"Connection established with {addr}")

# Continuously send a message to the client
try:
    while True:
        message = "Hello from the Python server!\n"
        conn.sendall(message.encode())  # Send message to client
        print(f"Sent message: {message}")
        time.sleep(2)  # Wait for 2 seconds before sending the next message

except KeyboardInterrupt:
    print("Server stopped.")

finally:
    # Close the connection when done
    conn.close()
