# globals.py
from zwdx.utils import getenv

# Flask + SocketIO objects
app = None
socketio = None

# Logger
logger = None

# Registered clients and jobs
registered_clients = []
job_results = {}

# Master port for distributed training
master_port = "29500"

# Flask host and port
FLASK_HOST = getenv("FLASK_HOST")
FLASK_PORT = int(getenv("FLASK_PORT"))
