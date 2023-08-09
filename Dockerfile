FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV LANG C.UTF-8

RUN set -x && \
    apt-get update -qq && \
    apt-get install --no-install-recommends -qq -y git build-essential python3-dev python-is-python3 ca-certificates wget && \
    apt-get install --no-install-recommends -qq -y ffmpeg libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

# if this is called "PIP_VERSION", pip explodes with "ValueError: invalid truth value '<VERSION>'"
ENV PYTHON_PIP_VERSION 23.2.1
# https://github.com/docker-library/python/issues/365
ENV PYTHON_SETUPTOOLS_VERSION 68.0.0
# https://github.com/pypa/get-pip
ENV PYTHON_GET_PIP_URL https://github.com/pypa/get-pip/raw/0d8570dc44796f4369b652222cf176b3db6ac70e/public/get-pip.py
ENV PYTHON_GET_PIP_SHA256 96461deced5c2a487ddc65207ec5a9cffeca0d34e7af7ea1afc470ff0d746207

RUN set -eux; \
	\
	wget -O get-pip.py "$PYTHON_GET_PIP_URL"; \
	echo "$PYTHON_GET_PIP_SHA256 *get-pip.py" | sha256sum -c -; \
	\
	export PYTHONDONTWRITEBYTECODE=1; \
	\
	python get-pip.py \
		--disable-pip-version-check \
		--no-cache-dir \
		--no-compile \
		"pip==$PYTHON_PIP_VERSION" \
		"setuptools==$PYTHON_SETUPTOOLS_VERSION" \
	; \
	rm -f get-pip.py; \
	\
	pip --version

RUN mkdir /app
ADD requirements*.txt /app
RUN set -x && \
    python -m pip install --no-cache-dir -r /app/requirements.txt && \
    python -m pip install --no-cache-dir "git+https://github.com/IDEA-Research/GroundingDINO.git#egg=groundingdino"

ADD . /app
WORKDIR /app

RUN set -xe && \
    useradd -s /bin/bash --uid 1000 --user-group -d /app app && \
    chown -R app:app /app

EXPOSE 1988

USER app

ENTRYPOINT ["python", "/app/main.py"]
