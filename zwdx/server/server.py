import logging
from flask import Flask
from flask_socketio import SocketIO
from threading import Lock

from zwdx.utils import getenv

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("server")

class Server:
    _instance = None
    _lock = Lock()
    
    def __init__(self, host=None, port=None, master_port="29500"):
        """Private constructor — use instance() to get the singleton."""
        if Server._instance is not None:
            raise RuntimeError("Server is a singleton. Use Server.instance() to access existing instance.")
        
        from zwdx.server.job import JobPool
        from zwdx.server.gpu_client_node import GPUClientPool
        from zwdx.server.room import RoomPool
        
        # Configuration
        self.FLASK_HOST = host or getenv("FLASK_HOST")
        self.FLASK_PORT = int(port or getenv("FLASK_PORT"))
        self.master_port = master_port
        
        # Logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)
        
        self.client_pool = GPUClientPool()
        self.room_pool = RoomPool()
        self.job_pool = JobPool()
        
        self.app = None
        self.socketio = None
        
        self.logger = logger
        
        Server._instance = self
    
    @classmethod
    def instance(cls):
        """Return the singleton instance (create if it doesn't exist)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset singleton instance. Useful for testing."""
        cls._instance = None
    
    def create_app(self):
        """Create and configure Flask application with SocketIO."""
        if self.app is not None:
            return self.app
        
        self.app = Flask(__name__)
        
        from flask_cors import CORS
        CORS(self.app, resources={r"/*": {"origins": "*"}})
        
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            logger=True,
            engineio_logger=True,
            ping_timeout=60,
            ping_interval=25,
            max_http_buffer_size=100_000_000,
        )
        
        # Register all routes
        self._register_routes()
        
        return self.app
    
    def _register_routes(self):
        """Register all application routes and socket handlers."""
        from zwdx.server.core_routes import register_core_routes
        from zwdx.server.job import register_job_routes
        from zwdx.server.room import register_room_routes

        register_core_routes()
        register_job_routes()
        register_room_routes()
    
    def run(self):
        """Start the Flask server with SocketIO."""
        if self.app is None:
            self.create_app()
        
        self.logger.info(f"Starting server on {self.FLASK_HOST}:{self.FLASK_PORT}")
        self.socketio.run(
            self.app, 
            host=self.FLASK_HOST, 
            port=self.FLASK_PORT, 
            allow_unsafe_werkzeug=True
        )