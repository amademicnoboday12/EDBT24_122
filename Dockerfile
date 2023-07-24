FROM continuumio/miniconda3:latest

RUN conda install -c intel mkl

COPY src /app/src
COPY resources/hamlet /data/hamlet

RUN for f in $(find /app/src -name requirements.txt); do echo "installing $f" && pip install -r $f; done

ENV PYTHONPATH="${PYTHONPATH}:/app/src/"

WORKDIR /app/src/

CMD ["python", "experiments/experiment.py"]
