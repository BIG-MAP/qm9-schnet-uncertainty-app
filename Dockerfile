FROM continuumio/miniconda3

WORKDIR /usr/src/app

RUN conda install -y numpy
RUN conda install -y pandas
RUN conda install -y pytorch cpuonly -c pytorch
RUN pip install scikit-learn==1.0
RUN pip install ase
