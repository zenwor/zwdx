#!/bin/bash

export PYTHONPATH="$(dirname "$PWD")"
export $(grep -v '^#' .env | xargs)