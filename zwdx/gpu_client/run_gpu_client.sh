#!/bin/bash

# Parse arguments
ROOM_TOKEN=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -rt|--room_token)
      ROOM_TOKEN="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 -rt <ROOM_TOKEN> or $0 --room_token <ROOM_TOKEN>"
      exit 1
      ;;
  esac
done

# Check if ROOM_TOKEN is provided
if [[ -z "$ROOM_TOKEN" ]]; then
  echo "Error: You must provide a room token with -rt or --room_token"
  echo "Usage: $0 -rt <ROOM_TOKEN> or $0 --room_token <ROOM_TOKEN>"
  exit 1
fi

echo "Using ROOM_TOKEN: $ROOM_TOKEN"

# Run Docker container
docker run --gpus all -it \
  -v ~/zwdx:/app \
  --add-host=host.docker.internal:host-gateway \
  -w /app \
  zwdx:latest \
  bash -c "cd zwdx && source ./setup.sh && cd gpu_client && pip install --root-user-action ignore dill && python3 gpu_client.py -rt $ROOM_TOKEN"
