FROM python:3.11-slim-bookworm AS signals-classification-base

ENV PYTHONUNBUFFERED 1

# Patch inherited base OS packages and install the runtime library scikit-learn
# needs (libgomp1, OpenMP). curl is only used to fetch the Poetry installer, so
# it is removed again in the same layer to keep it out of the final image.
RUN set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install -y --no-install-recommends \
        curl \
        libgomp1; \
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3; \
    cd /usr/local/bin; \
    ln -s /opt/poetry/bin/poetry; \
    poetry config virtualenvs.create false; \
    poetry completions bash >> ~/.bash_completion; \
    poetry self add poetry-plugin-sort; \
    apt-get purge -y --auto-remove curl; \
    rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app


FROM signals-classification-base AS signals-classification-web

# uWSGI ships no wheel and is compiled here. The build toolchain is removed in
# the same layer, leaving only the libpcre3 runtime library uWSGI links against.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpcre3 \
        libpcre3-dev; \
    poetry install --with web; \
    apt-get purge -y --auto-remove build-essential libpcre3-dev; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/app

ENV UWSGI_HTTP :8000
ENV UWSGI_MODULE app:application
ENV UWSGI_PROCESSES 1
ENV UWSGI_THREADS 4
ENV UWSGI_MASTER 1
ENV UWSGI_OFFLOAD_THREADS 1
ENV UWSGI_HARAKIRI 25

CMD ["uwsgi"]


FROM signals-classification-base AS signals-classification-train

ENV NLTK_DATA /usr/local/share/nltk_data

RUN poetry install --with train

RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords

ENTRYPOINT ["python", "/app/app/train/run.py"]
