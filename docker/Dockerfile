FROM cr.ai.nebius.cloud/crnf2coti090683j5ssi/kernels/default:2024-09-23-17-29-46-81de2bdf

USER root
RUN python3 -m pip install ytsaurus-client ytsaurus-yson
RUN python3 -m pip install ytsaurus-rpc-driver
RUN pip install --upgrade pip

RUN apt update
RUN apt install libopenmpi-dev libmpich-dev -y

COPY ./gpt-neox-tractoai/requirements/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./gpt-neox-tractoai/requirements/requirements-wandb.txt requirements-wandb.txt
RUN pip install -r requirements-wandb.txt

RUN pip install -U tractorun

RUN pip install setuptools==69.5.1
RUN pip install deepspeed==0.14.4

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-4

ENV PATH /usr/local/cuda-12.4/bin:$PATH
ENV CUDA_HOME /usr/local/cuda-12.4

COPY gai.conf /etc/gai.conf

USER 1000

ENV PATH /usr/local/cuda-12.4/bin:$PATH
ENV CUDA_HOME /usr/local/cuda-12.4

