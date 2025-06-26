FROM python:3.12-slim AS builder

ARG REQUIREMENTS=requirements.txt
COPY ${REQUIREMENTS} requirements.txt
RUN pip install --no-cache -r requirements.txt

FROM python:3.12-slim


COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy and install dependencies
# Copy dependency files first for caching
COPY pyproject.toml ./

# Copy app code
COPY app/ .

RUN mkdir -p s3

CMD ["python", "main.py"]
