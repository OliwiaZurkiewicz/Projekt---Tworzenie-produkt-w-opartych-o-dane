# Projekt - Tworzenie produktów opartych na danych
Repozytorium kodu projektu aplikacji Dash analizy i prognozowania cen mieszkań przy użyciu sztucznej inteligencji.

## Uruchomienie 
Aplikację internetową można uruchomić jedną z komend make:
```bash
$ make run-docker
$ make run
```

Lub bezpośrednio przy użyciu Dockera:
```bash
$ docker compose -f dockerfiles/docker-compose.yaml up application
```

Lub bezpośrednio z poziomu systemu macOS:
```bash
$ brew install micromamba \
  && micromamba create --name produkty python=3.12 --channel conda-forge \
  && micromamba activate produkty \
  && pip install poetry \
  && poetry install --no-root \
  && python3 Application.py
```

Lub systemu Linux:
```bash
$ apt update && apt install micromamba \
  && micromamba create --name produkty python=3.12 --channel conda-forge \
  && micromamba activate produkty \
  && pip install poetry \
  && poetry install --no-root \
  && python3 Application.py
```

## Testy
Testy jednostkowe można uruchomić jednym z poleceń make:
```bash
$ make test-docker
$ make test
```

## Aplikacja
Aplikacja jest dostępna po uruchomieniu pod adresem: http://127.0.0.1:8080