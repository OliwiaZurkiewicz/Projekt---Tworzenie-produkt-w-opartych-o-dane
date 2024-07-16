#!/bin/bash

echo 'RUNNING AUTOFLAKE...'
autoflake --verbose --recursive --expand-star-imports --remove-all-unused-imports --in-place .

echo -e '\nRUNNING BANDIT...'
bandit -r --config pyproject.toml .

echo -e '\nRUNNING ISORT...'
isort --settings-path pyproject.toml --profile black .

echo -e '\nRUNNING BLACK...'
black --config pyproject.toml .

echo -e '\nRUNNING MYPY...'
mypy .
