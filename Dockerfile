FROM python:3.11-slim as builder

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git

COPY requirements.txt /requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --no-warn-script-location --user -r requirements.txt

# Stage 2: Runtime
FROM tensorflow/tensorflow:2.13.0-gpu
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV TZ=America/New_York

RUN apt update && \
    apt install --no-install-recommends -y libgl1-mesa-glx && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY --from=builder /root/.local/lib/python3.11/site-packages /usr/local/lib/python3.11/dist-packages
COPY app.py app.py

CMD ["python3", "-u", "app.py"]
EXPOSE 7860
