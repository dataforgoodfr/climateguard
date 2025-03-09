import logging
import os

def get_secret_docker(secret_name):
    secret_value: str = os.environ.get(secret_name, None)

    if secret_value and os.path.exists(secret_value):
        with open(secret_value, "r") as file:
            return file.read().strip()
    return secret_value