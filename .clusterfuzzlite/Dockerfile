FROM gcr.io/oss-fuzz-base/base-builder-python

COPY . /src/edgeml-python
WORKDIR /src/edgeml-python

RUN pip install atheris
RUN pip install -e edgeml/python[all] || pip install -e edgeml/python

COPY .clusterfuzzlite/build.sh /src/build.sh
