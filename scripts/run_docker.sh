#!/bin/bash

docker compose -f dockerfiles/docker-compose.yaml up application --build --force-recreate
