FROM gcr.io/oss-fuzz-base/base-builder-python

COPY . /src/octomil-python
WORKDIR /src/octomil-python

RUN pip install atheris
RUN pip install -e octomil/python[all] || pip install -e octomil/python

COPY .clusterfuzzlite/build.sh /src/build.sh
