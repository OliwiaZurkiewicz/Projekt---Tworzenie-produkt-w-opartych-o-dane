.PHONY: run run-docker test test-docker

lint:
	@./lint.sh

run:
	@scripts/run.sh

run-docker:
	@scripts/run_docker.sh

test:
	@scripts/run_tests.sh

test-docker:
	@scripts/run_tests_docker.sh
