FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:0a7f095be949c6168f62f51bfdda1e8d8e7fb390afec1ceb601948bc7d5367bb

COPY . /src/edgeml-python
WORKDIR /src/edgeml-python

RUN pip install atheris
RUN pip install -e edgeml/python[all] || pip install -e edgeml/python

COPY .clusterfuzzlite/build.sh /src/build.sh
