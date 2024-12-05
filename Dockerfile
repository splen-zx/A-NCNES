# syntax=docker/dockerfile:1
FROM ubuntu:22.04
WORKDIR /project
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y openssh-server curl && \
    echo 'root:root' | chpasswd && \
    echo -n "PermitRootLogin yes" | cat >> /etc/ssh/sshd_config && \
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o ~/get-pip.py && \
    python3 ~/get-pip.py && \
    python3 -m pip install --upgrade pip setuptools wheel  --root-user-action ignore


CMD sh -c "service ssh restart;/bin/bash"

# already with python 3.10.12
# tag ubuntu-22.04:ssh
# docker run -it -p 16622:22 -p 16680:8888 --restart=always --gpus all ubuntu-22.04:ssh
