#!/bin/bash

# Start the zwdx server
echo "Starting zwdx server on port $FLASK_PORT..."
uv run python3 app.py

sleep 2

# Start LocalTunnel
# if [ -n "$LT_SUBDOMAIN" ]; then
#     echo "Starting LocalTunnel on port $FLASK_PORT with subdomain $LT_SUBDOMAIN..."
#     lt --port $FLASK_PORT --subdomain $LT_SUBDOMAIN &
#     LT_PID=$!
# else
#     echo "Starting LocalTunnel on port $FLASK_PORT (random subdomain)..."
#     lt --port $FLASK_PORT &
#     LT_PID=$!
# fi

cleanup() {
    echo "Stopping zwdx server..."
    # uv will terminate automatically with Ctrl+C
    # Optional: forcibly free the port
    if lsof -i :$FLASK_PORT >/dev/null; then
        fuser -k $FLASK_PORT/tcp
    fi
    exit
}

trap cleanup SIGINT SIGTERM

echo "Starting zwdx 