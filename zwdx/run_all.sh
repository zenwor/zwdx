#!/bin/bash

mkdir -p logs

kill_port() {
    PORT=$1
    if lsof -i:$PORT -t >/dev/null; then
        echo "Port $PORT is in use. Killing existing process..."
        lsof -ti:$PORT | xargs kill -9
    fi
}

kill_port 5561   # DB
kill_port 4461   # Server
kill_port 3000   # UI

npx concurrently \
  --names "DB,Server,UI" \
  --prefix "[{name}]" \
  --kill-others \
  --success first \
  \
  "bash -c 'cd server/db && ./run_db.sh | tee ../../logs/db.log'" \
  "bash -c 'cd server && ./run_server.sh | tee ../logs/server.log'" \
  "bash -c 'cd ui && npm start | tee ../logs/ui.log'"
