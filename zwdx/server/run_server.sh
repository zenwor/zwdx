# #!/bin/bash

# # Start the zwdx server
# echo "Starting zwdx server on port $FLASK_PORT..."
# uv run python3 app.py &
# SERVER_PID=$!

# # Function to clean up on exit
# cleanup() {
#     echo "Stopping zwdx server..."
#     kill $SERVER_PID 2>/dev/null

#     # Free the port if still in use
#     if lsof -i :$FLASK_PORT >/dev/null; then
#         echo "Port $FLASK_PORT is still in use. Killing process..."
#         fuser -k $FLASK_PORT/tcp
#     fi

#     exit
# }

# # Catch termination signals
# trap cleanup SIGINT SIGTERM

# # Keep script running
# wait $SERVER_PID

# # ----------------------------------------
# # LT (localtunnel) section commented for now
# # echo "Starting lt on port $LT_PORT with subdomain $LT_SUBDOMAIN..."
# # lt --port $LT_PORT --subdomain $LT_SUBDOMAIN
# # cleanup

#!/bin/bash

# Start the zwdx server
echo "Starting zwdx server on port $FLASK_PORT..."
uv run python3 app.py &
SERVER_PID=$!

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

# Function to clean up on exit
cleanup() {
    echo "Stopping zwdx server..."
    kill $SERVER_PID 2>/dev/null
    if [ -n "$LT_PID" ]; then
        echo "Stopping LocalTunnel..."
        kill $LT_PID 2>/dev/null
    fi

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