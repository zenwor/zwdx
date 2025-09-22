from flask import Flask, jsonify
from zwdx.utils import getenv

FLASK_HOST = getenv("FLASK_HOST")
FLASK_PORT = int(getenv("FLASK_PORT"))

app = Flask(__name__)

@app.route("/connect")
def connect():
    return jsonify({"status": "connected"})

if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT)