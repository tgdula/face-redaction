FROM python:3.11-slim

RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

RUN poetry install  --no-interaction --no-ansi

COPY ./face_redaction ./face_redaction

RUN poetry install --no-interaction --no-ansi

ENTRYPOINT ["poetry", "run", "redact"]