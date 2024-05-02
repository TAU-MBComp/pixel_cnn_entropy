FROM ubuntu:24.04
RUN apt update -y && apt install -y cmake vim
RUN apt install -y wget build-essential checkinstall  libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
RUN cd /usr/src && wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz && tar xzf Python-3.8.10.tgz
RUN cd /usr/src/Python-3.8.10 && ./configure --enable-optimizations
RUN cd /usr/src/Python-3.8.10 && make -j 16
RUN cd /usr/src/Python-3.8.10 && make install

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN python3 -m pip install -r requirements.txt
