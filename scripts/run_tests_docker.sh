#!/bin/bash

docker compose -f dockerfiles/docker-compose.yaml up application_test --build --force-recreate