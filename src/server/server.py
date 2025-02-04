import socketserver
import threading
import datetime

class StreamRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        print(f"Connection from: {self.client_address}")
        
        while True:
            try:
                data = self.request.recv(1024)
                if not data:
                    print(f"Connection closed: {self.client_address}")
                    break
                
                message = data.decode('utf-8')
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] Message received from {self.client_address}: {message}")
                
            except ConnectionResetError:
                print(f"Connection reset by peer: {self.client_address}")
                break
            except Exception as e:
                print(f"Error: {e}")
                break

class StreamServer:
    def __init__(self, host='0.0.0.0', port=9000):
        self.server = socketserver.ThreadingTCPServer((host, port), StreamRequestHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        print("Starting server...")
        self.server_thread.start()
        print(f"Server running on {self.server.server_address}")

    def stop(self):
        print("Stopping server...")
        self.server.shutdown()
        self.server.server_close()
        print("Server stopped.")

if __name__ == "__main__":
    server = StreamServer(port=9000)
    try:
        server.start()
        while True:
            pass
    except KeyboardInterrupt:
        server.stop()
