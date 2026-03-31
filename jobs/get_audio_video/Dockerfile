FROM python:3.13-slim

ARG REQUIREMENTS=requirements.txt
COPY ${REQUIREMENTS} requirements.txt
RUN pip install --no-cache -r requirements.txt

WORKDIR /app

COPY get_audio_video.py .

ENTRYPOINT ["python", "get_audio_video.py"]