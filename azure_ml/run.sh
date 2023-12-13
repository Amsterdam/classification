#!/usr/bin/env bash

if [ -z "$1" ]; then
  echo "Please specify the path to the CSV file!"
  exit 1
fi

if [ ! -f .env ]; then
  echo "Please copy .env.example to .env and modify it to have the correct values!"
  exit 2
fi

source .env

az ml job create \
--file pipeline.yaml \
-g "${GROUP}" \
-w "${WORKSPACE}" \
--set settings.default_compute="${COMPUTE}" \
--set inputs.csv.path="$1" \
--web
