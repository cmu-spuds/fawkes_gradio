FROM continuumio/miniconda3
EXPOSE 7860

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx && \
    apt-get install -y --no-install-recommends libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV GRADIO_SERVER_NAME=0.0.0.0
WORKDIR /workspace

ADD environment.yaml /workspace/environment.yaml
RUN conda env update -n base --file environment.yaml && \
    pip install typing-extensions -U

ADD app.py /workspace/
CMD [ "python" , "/workspace/app.py" ]
