FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
RUN mkdir /workspace
ARG http_proxy
ARG https_proxy
WORKDIR /workspace
RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt update && \
    apt install -y --no-install-recommends \
        git \
        wget \
        curl \
        build-essential \
        vim \
        python3 \
        python3-dev \
        python \
        python-dev \
        libgl1-mesa-dev \
        libgtk2.0-dev \
        freeglut3-dev \
        pyqt4-dev-tools
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN python get-pip.py

# OpenGL
RUN dpkg --add-architecture i386 && \
    apt update && \
    apt install -y --no-install-recommends \
        libxau6 libxau6:i386 \
        libxdmcp6 libxdmcp6:i386 \
        libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 \
        libx11-6 libx11-6:i386
RUN rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES \
        ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
        ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

## Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt update && \
    apt install -y --no-install-recommends \
    	libglvnd0 libglvnd0:i386 \
        libgl1 libgl1:i386 \
        libglx0 libglx0:i386 \
        libegl1 libegl1:i386 \
        libgles2 libgles2:i386
RUN rm -rf /var/lib/apt/lists/*

RUN echo \
'{\n\
    "file_format_version": "1.0.0",\n\
    "ICD": {\n\
        "library_path": "libEGL_nvidia.so.0"\n\
    }\n\
}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt update && \
    apt install -y --no-install-recommends \
        pkg-config \
    	libglvnd-dev libglvnd-dev:i386 \
        libgl1-mesa-dev libgl1-mesa-dev:i386 \
        libegl1-mesa-dev libegl1-mesa-dev:i386 \
        libgles2-mesa-dev libgles2-mesa-dev:i386
RUN rm -rf /var/lib/apt/lists/*

RUN mkdir temp && cd temp
COPY . .
RUN pip install -r requirements.txt && \
    pip install --user -e .
RUN cd .. && rm -rf temp
RUN sed -i 's/device: EGLDevice = None/device = None/g' \
    /usr/local/lib/python2.7/dist-packages/pyrender/platforms/egl.py
