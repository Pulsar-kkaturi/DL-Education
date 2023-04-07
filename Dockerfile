# Based on nvidia/cuda
FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04

# Linux Environment Setting
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install tzdata
ENV TZ=Asia/Seoul
RUN apt-get update && apt-get -y install \
    vim pkg-config libcairo2 libcairo2-dev \
    git sudo gcc ssh zsh openssh-server tmux \
    python3-dev python3-pip python3-setuptools \
    ca-certificates software-properties-common hdf5-tools 
# openslide-tools python3-openslide 
RUN pip3 install --upgrade pip
RUN service ssh start

# User setting
RUN adduser student
#RUN groupadd -g ${GROUP_ID} lungteam
#RUN useradd -u ${USER_ID} -g lungteam -p $(openssl passwd -1 vuno2018) vuno
RUN echo "student ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER student
WORKDIR /home/student

# Python Setting
ENV PATH="/home/student/.local/bin:${PATH}"
RUN sudo ln -s /usr/bin/python3 /usr/bin/python
RUN sudo cp /usr/local/bin/pip* /usr/bin/.
ENV PYTHONPATH "/home/student/Projects:${PYTHONPATH}"
# Python library Installation (etc. openslide-python)
RUN pip3 install --upgrade tensorflow
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install numpy scipy pandas pillow tqdm scikit-learn scikit-image matplotlib ipykernel opencv-python-headless
RUN pip3 install psf perlin-noise tifffile seaborn 
RUN pip3 install pydicom SimpleITK
COPY --chown=student:student requirements.txt .
# RUN pip3 install -r requirements.txt
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN zsh