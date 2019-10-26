FROM python:3.7.5
LABEL MAINTAINER="Kevin" 

# linux packages
RUN apt-get update && apt-get upgrade -y --force-yes
RUN apt-get install -y --force-yes software-properties-common 
RUN apt-add-repository -r ppa:armagetronad-dev/ppa
RUN apt-get update && apt-get upgrade -y --force-yes
RUN apt-get install -y --force-yes build-essential libssl-dev libffi-dev python-dev python-pip libsasl2-dev libldap2-dev expect vim python-numpy gdal-bin libgdal-dev

# python dependency
RUN ["pip3","--default-timeout=300 ", "install","pystan"] 
COPY requirements.txt requirements.txt
RUN ["pip3","--default-timeout=300 ", "install","-r","requirements.txt"]

COPY Makefile /opt/Makefile
COPY config.py /opt/config.py
COPY run.py /opt/src/run.py
COPY src /opt/src

WORKDIR /opt
LABEL version="1.0" \
      description="This image is used to set up data flow service."