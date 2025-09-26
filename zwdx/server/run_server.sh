#!/bin/bash

# Start the zwdx server
echo "Starting zwdx server on port $FLASK_PORT..."
uv run python3 app.py &
SERVER_PID=$!

# Function to clean up on exit
cleanup() {
    echo "Stopping zwdx server..."
    kill $SERVER_PID 2>/dev/null

    # Free the port if still in use
    if lsof -i :$FLASK_PORT >/dev/null; then
        echo "Port $FLASK_PORT is still in use. Killing process..."
        fuser -k $FLASK_PORT/tcp
    fi

    exit
}

# Catch termination signals
trap cleanup SIGINT SIGTERM

# Keep script running
wait $SERVER_PID

# ----------------------------------------
# LT (localtunnel) section commented for now
# echo "Starting lt on port $LT_PORT with subdomain $LT_SUBDOMAIN..."
# lt --port $LT_PORT --subdomain $LT_SUBDOMAIN
# cleanup