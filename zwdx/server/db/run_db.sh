#!/bin/bash

if [ ! -d "$MONGODB_DBPATH" ]; then
  echo "Creating MongoDB data directory at $MONGODB_DBPATH..."
  mkdir -p "$MONGODB_DBPATH"
fi

echo "Starting MongoDB on port $MONGODB_PORT..."
mongod --port "$MONGODB_PORT" --dbpath "$MONGODB_DBPATH" --quiet &
MONGO_PID=$!

cleanup() {
  echo ""
  echo "Stopping MongoDB (PID: $MONGO_PID)..."
  kill $MONGO_PID
  wait $MONGO_PID 2>/dev/null
  echo "MongoDB stopped."
  exit 0
}

trap cleanup SIGINT SIGTERM

echo "MongoDB is running (PID: $MONGO_PID). Press Ctrl+C to stop."
wait $MONGO_PID
