FROM python:3.12-slim

WORKDIR /app

# Copy and install dependencies
# Copy dependency files first for caching
COPY pyproject.toml ./
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy app code
COPY app/ .

RUN mkdir -p s3

CMD ["python", "main.py"]
