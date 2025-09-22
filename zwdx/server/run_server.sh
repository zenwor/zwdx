#!/bin/bash

echo "Starting zwdx server on port $FLASK_PORT..."
uv run python3 server.py &

SERVER_PID=$!

cleanup() {
    echo "Stopping zwdx server..."
    kill $SERVER_PID
    exit
}

trap cleanup SIGINT SIGTERM

echo "Starting lt on port $LT_PORT with subdomain $LT_SUBDOMAIN..."
lt --port $LT_PORT --subdomain $LT_SUBDOMAIN

cleanup
