FROM python:3.12.4-slim AS runtime

USER root

RUN apt update -y && apt install git-lfs -y

COPY ./pyproject.toml /opt/project/

WORKDIR /opt/project

RUN pip install poetry

RUN --mount=type=cache,target=/root/.cache/pypoetry/cache \
    --mount=type=cache,target=/root/.cache/pypoetry/artifacts \
    poetry config virtualenvs.create false \
    && poetry lock --no-update \
    && poetry install --without dev --no-ansi --no-interaction --no-root

COPY ./src/main.py /opt/project/
COPY ./model_DT.pkl /opt/project/
COPY ./model_FR.pkl /opt/project/
COPY ./raw_sales.csv /opt/project/

USER 10000
ENV PYTHONPATH=/opt/project

EXPOSE 8080/tcp

CMD ["python3", "main.py"]

FROM runtime AS test

WORKDIR /opt/project

USER root

RUN --mount=type=cache,target=/root/.cache/pypoetry/cache \
    --mount=type=cache,target=/root/.cache/pypoetry/artifacts \
    poetry config virtualenvs.create false \
    && poetry install --no-ansi --no-interaction --no-root

COPY ./test_raw_sales.csv /opt/project/
COPY ./tests/unit_test.py /opt/project/

CMD ["pytest", "-v", "."]
