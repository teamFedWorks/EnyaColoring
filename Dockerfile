# # Use a base image with CUDA 13.0 and cuDNN
# FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=America/New_York

# ENV PATH="/usr/local/cuda/bin:${PATH}"
# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# # Timezone & basic tools
# RUN apt-get update && apt-get install -y tzdata && \
#     ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

# # System dependencies
# RUN apt-get update && apt-get install -y \
#     git cmake=3.28.3-1build7 build-essential=12.10ubuntu1 pkg-config=1.8.1-2build1 libprotobuf-dev=3.21.12-8.2ubuntu0.2 libleveldb-dev=1.23-5build1 libsnappy-dev=1.1.10-1build1 \
#     libopencv-dev=4.6.0+dfsg-13.1ubuntu1 libhdf5-serial-dev protobuf-compiler=3.21.12-8.2ubuntu0.2 libgflags-dev=2.2.2-2build1 \
#     liblmdb-dev=0.9.31-1build1 \
#     libboost-all-dev=1.83.0.1ubuntu2 wget=1.21.4-1ubuntu4.1 curl=8.5.0-2ubuntu10.6 unzip=6.0-28ubuntu4.1 nano=7.2-2ubuntu0.1 wget=1.21.4-1ubuntu4.1 \
#     curl=8.5.0-2ubuntu10.6 \
#     git=1:2.43.0-1ubuntu7.3 \
#     build-essential=12.10ubuntu1 g++=4:13.2.0-7ubuntu1 pkg-config \
#     libass-dev=1:0.17.1-2build1 \
#     libfdk-aac-dev=2.0.2-3~ubuntu4 \
#     librtmp-dev=2.4+20151223.gitfa8646d.1-2build7 \
#     libssl-dev=3.0.13-0ubuntu3.6 \
#     libvdpau1=1.5-2build1 \
#     libvorbis-dev=1.3.7-1build3 \
#     libvpx-dev=1.14.0-1ubuntu2.2 \
#     libx264-dev=2:0.164.3108+git31e19f9-1 \
#     libx265-dev=3.5-2build1 \
#     ffmpeg=7:6.1.1-3ubuntu5 mplayer=2:1.5+svn38446-1build5 mencoder=2:1.5+svn38446-1build5 && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# # Install Miniconda
# ENV CONDA_DIR /opt/conda
# ENV PATH=$CONDA_DIR/bin:$PATH
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
#     /bin/bash miniconda.sh -b -p /opt/conda && \
#     rm miniconda.sh


# # Accept the Conda Terms of Service for default channels
# RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
#     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ENV CONDA_DEFAULT_ENV=dl_env


# # Create a new conda environment with Python 3.11
# RUN conda create -n ${CONDA_DEFAULT_ENV} python=3.11 -y && \
#     conda run -n ${CONDA_DEFAULT_ENV} pip install --no-cache-dir --upgrade pip




# # Python dependencies like Torch and SAM2
# RUN conda install -n ${CONDA_DEFAULT_ENV} nvidia/label/cuda-13.0.1::cuda-toolkit conda-forge::cudnn=9.13.1
# # RUN conda run -n ${CONDA_DEFAULT_ENV} pip install --no-cache-dir \
# #     torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
# RUN conda run -n ${CONDA_DEFAULT_ENV} pip install --no-cache-dir \
#     torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# # Diffusers with PyTorch
# RUN conda run -n ${CONDA_DEFAULT_ENV} pip install  "diffusers[torch]"

# # PEFT (from PyPI)
# RUN conda run -n ${CONDA_DEFAULT_ENV} pip install  "peft"

# RUN conda run -n ${CONDA_DEFAULT_ENV} pip install  tensorflow[and-cuda] keras==3.11.3 keras-cv==0.9.0
# RUN conda run -n ${CONDA_DEFAULT_ENV} pip install  transformers==4.57.0 matplotlib==3.10.7 jupyter==1.1.1 notebook==7.4.7 pillow==11.0.0 pydot==4.0.1 scipy==1.16.2 \
#     jupyterlab==4.4.9 ipykernel==6.30.1 ipython==9.6.0 python-dateutil==2.9.0.post0 mediapipe==0.10.21 ultralytics==8.3.207 supervision==0.26.1 \
#     fastai==2.8.4 opencv-python-headless==4.11.0.86 scikit-learn==1.7.2 yapf==0.43.0 black==25.9.0 isort==6.1.0 jupyterlab_code_formatter==3.0.2


# ENV SAM_ROOT=/opt/sam2
# WORKDIR $SAM_ROOT
# RUN git clone https://github.com/teamFedWorks/sam2.git ${SAM_ROOT}
# RUN conda run -n ${CONDA_DEFAULT_ENV} pip install -e ${SAM_ROOT}

# ENV PYTHONPATH="${SAM_ROOT}:$PYTHONPATH"
# RUN mkdir -p $SAM_ROOT/checkpoints && \
#     wget -O $SAM_ROOT/checkpoints/sam2.1_hiera_large.pt \
#     https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt

# ENV DEOLDIFY_ROOT=/opt/deoldify
# WORKDIR ${DEOLDIFY_ROOT}
# RUN git clone https://github.com/teamFedWorks/DeOldify.git ${DEOLDIFY_ROOT}
# COPY visualize.py ${DEOLDIFY_ROOT}/deoldify/visualize.py
# RUN conda run -n ${CONDA_DEFAULT_ENV} pip install yt-dlp ffmpeg-python ffmpeg pillow scikit-image realesrgan basicsr onnxruntime-gpu
# RUN sed -i 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/g' \
#     /opt/conda/envs/dl_env/lib/python3.11/site-packages/basicsr/data/degradations.py
# ENV PYTHONPATH="${DEOLDIFY_ROOT}:$PYTHONPATH"



# # --- DeepEx ---
# ENV DEEPEX_ROOT=/opt/deepex
# WORKDIR ${DEEPEX_ROOT}
# RUN git clone https://github.com/teamFedWorks/ComfyUI-Deep-Exemplar-based-Video-Colorization.git ${DEEPEX_ROOT}
# COPY colorize_video.py ${DEEPEX_ROOT}/colorize_video.py
# ENV PYTHONPATH="${DEEPEX_ROOT}:$PYTHONPATH"



# # COMFY UI SETUP
# ENV COMFYUI_ROOT=/opt/comfyui
# RUN git clone https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_ROOT}
# RUN conda run -n ${CONDA_DEFAULT_ENV} pip install --no-cache-dir -r ${COMFYUI_ROOT}/requirements.txt
# ENV PYTHONPATH="${COMFYUI_ROOT}:$PYTHONPATH"



# # Set root directory
# ENV DEEPREMASTER_ROOT=/opt/deepremaster
# WORKDIR ${DEEPREMASTER_ROOT}
# RUN git clone https://github.com/teamFedWorks/siggraphasia2019_remastering.git ${DEEPREMASTER_ROOT}
# # COPY remaster.py ${DEEPREMASTER_ROOT}/remaster.py
# ENV PYTHONPATH="${DEEPREMASTER_ROOT}:$PYTHONPATH"


# # --- GFPGAN setup ---
# ENV GFPGAN_ROOT=/opt/gfpgan
# WORKDIR ${GFPGAN_ROOT}
# RUN git clone https://github.com/TencentARC/GFPGAN.git ${GFPGAN_ROOT}
# # Add GFPGAN to PYTHONPATH
# COPY inference_gfpgan.py ${GFPGAN_ROOT}/inference_gfpgan.py
# ENV PYTHONPATH="${GFPGAN_ROOT}:$PYTHONPATH"



# # Install Python dependencies
# RUN conda run -n ${CONDA_DEFAULT_ENV} pip install -U timm==1.0.20 tqdm==4.67.1 ipywidgets==8.1.7 jupyterlab==4.4.9 scikit-image==0.25.2 kornia==0.8.1 plotly==6.3.1
# RUN conda install -n ${CONDA_DEFAULT_ENV} conda-forge::libsqlite --force-reinstall
# RUN conda run -n ${CONDA_DEFAULT_ENV} pip uninstall -y nvidia-cudnn-cu12 tb-nightly

# COPY requirements.txt .
# RUN conda run -n ${CONDA_DEFAULT_ENV} pip install -r requirements.txt
# RUN conda run -n dl_env pip install flask

# RUN conda init && \
#     echo "conda activate ${CONDA_DEFAULT_ENV}" >> ~/.bashrc



# # COPY workflow_api.py ${COMFYUI_ROOT}/workflow_api.py
# # COPY run_workflow.py ${COMFYUI_ROOT}/run_workflow.py
# # Jupyter runtime
# WORKDIR /workspace

# COPY entrypoint /entrypoint.sh
# RUN chmod +x /entrypoint.sh
# EXPOSE 8888 8188
# ENTRYPOINT ["/entrypoint.sh"]












































FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Timezone & basic tools
RUN apt-get update && apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

# System dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential pkg-config libprotobuf-dev libleveldb-dev libsnappy-dev \
    libopencv-dev libhdf5-serial-dev protobuf-compiler libgflags-dev \
    liblmdb-dev \
    libboost-all-dev wget curl unzip nano wget \
    curl \
    git \
    build-essential g++ pkg-config \
    libass-dev \
    libfdk-aac-dev \
    librtmp-dev \
    libssl-dev \
    libvdpau1 \
    libvorbis-dev \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    ffmpeg mplayer mencoder && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p ${CONDA_DIR} && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=dl_env

# Create a new conda environment with Python 3.11
RUN conda create -n ${CONDA_DEFAULT_ENV} python=3.11 -y && \
    conda run -n ${CONDA_DEFAULT_ENV} pip install --no-cache-dir --upgrade pip

# Python dependencies like Torch and SAM2
RUN conda install -n ${CONDA_DEFAULT_ENV} nvidia/label/cuda-12.2.0::cuda-toolkit conda-forge::cudnn=9.3
RUN conda run -n ${CONDA_DEFAULT_ENV} pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Diffusers with PyTorch
RUN conda run -n ${CONDA_DEFAULT_ENV} pip install --no-cache-dir "diffusers[torch]"

# PEFT (from PyPI)
RUN conda run -n ${CONDA_DEFAULT_ENV} pip install --no-cache-dir "peft"

RUN conda run -n ${CONDA_DEFAULT_ENV} pip install --no-cache-dir tensorflow[and-cuda] keras keras-cv
RUN conda run -n ${CONDA_DEFAULT_ENV} pip install --no-cache-dir transformers matplotlib jupyter notebook pillow pydot scipy \
    notebook jupyter jupyterlab ipykernel ipython python-dateutil mediapipe ultralytics supervision \
    fastai opencv-python-headless scikit-learn yapf black isort jupyterlab_code_formatter

ENV SAM_ROOT=/opt/sam2
WORKDIR $SAM_ROOT
RUN git clone https://github.com/facebookresearch/sam2.git ${SAM_ROOT}
RUN conda run -n ${CONDA_DEFAULT_ENV} pip install -e ${SAM_ROOT}

ENV PYTHONPATH="${SAM_ROOT}:$PYTHONPATH"
RUN mkdir -p $SAM_ROOT/checkpoints && \
    wget -O $SAM_ROOT/checkpoints/sam2.1_hiera_large.pt \
    https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt

ENV DEOLDIFY_ROOT=/opt/deoldify
WORKDIR ${DEOLDIFY_ROOT}
RUN git clone https://github.com/jantic/DeOldify.git ${DEOLDIFY_ROOT}
COPY visualize.py ${DEOLDIFY_ROOT}/deoldify/visualize.py
RUN conda run -n ${CONDA_DEFAULT_ENV} pip install yt-dlp ffmpeg-python ffmpeg pillow scikit-image realesrgan basicsr onnxruntime-gpu
RUN sed -i 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/g' \
    /opt/conda/envs/dl_env/lib/python3.11/site-packages/basicsr/data/degradations.py
ENV PYTHONPATH="${DEOLDIFY_ROOT}:$PYTHONPATH"



# --- DeepEx ---
ENV DEEPEX_ROOT=/opt/deepex
WORKDIR ${DEEPEX_ROOT}
RUN git clone https://github.com/jonstreeter/ComfyUI-Deep-Exemplar-based-Video-Colorization.git ${DEEPEX_ROOT}
COPY colorize_video.py ${DEEPEX_ROOT}/colorize_video.py
ENV PYTHONPATH="${DEEPEX_ROOT}:$PYTHONPATH"


# --- ComfyUI Setup (cu121-compatible) ---
# ENV COMFYUI_ROOT=/opt/comfyui
# WORKDIR ${COMFYUI_ROOT}
# RUN git clone https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_ROOT}
# # Use existing cu121 PyTorch, install only ComfyUI's other dependencies
# # RUN sed -i 's/comfyui-frontend-package==1.25.9/comfyui-frontend-package==1.25.8/' ${COMFYUI_ROOT}/requirements.txt && \
# #     sed -i 's/comfyui-workflow-templates==0.1.60/comfyui-workflow-templates==0.1.59/' ${COMFYUI_ROOT}/requirements.txt && \
# #     sed -i 's/comfyui-embedded-docs==0.2.6/comfyui-embedded-docs==0.2.4/' ${COMFYUI_ROOT}/requirements.txt

# RUN conda run -n ${CONDA_DEFAULT_ENV} pip install --no-cache-dir -r ${COMFYUI_ROOT}/requirements.txt
# # Copy your custom workflow.py

# ENV PYTHONPATH="${COMFYUI_ROOT}:$PYTHONPATH"


ENV COMFYUI_ROOT=/opt/comfyui
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_ROOT}
RUN conda run -n ${CONDA_DEFAULT_ENV} pip install --no-cache-dir -r ${COMFYUI_ROOT}/requirements.txt
ENV PYTHONPATH="${COMFYUI_ROOT}:$PYTHONPATH"



# Set root directory
ENV DEEPREMASTER_ROOT=/opt/deepremaster
WORKDIR ${DEEPREMASTER_ROOT}
RUN git clone https://github.com/satoshiiizuka/siggraphasia2019_remastering.git ${DEEPREMASTER_ROOT}
# COPY remaster.py ${DEEPREMASTER_ROOT}/remaster.py
ENV PYTHONPATH="${DEEPREMASTER_ROOT}:$PYTHONPATH"



# Install Python dependencies
RUN conda run -n ${CONDA_DEFAULT_ENV} pip install -U timm tqdm ipywidgets jupyterlab scikit-image kornia plotly
RUN conda install -n ${CONDA_DEFAULT_ENV} conda-forge::libsqlite --force-reinstall
RUN conda run -n ${CONDA_DEFAULT_ENV} pip uninstall -y nvidia-cudnn-cu12 tb-nightly

COPY requirements.txt .
RUN conda run -n ${CONDA_DEFAULT_ENV} pip install -r requirements.txt



RUN conda init && \
    echo "conda activate ${CONDA_DEFAULT_ENV}" >> ~/.bashrc

# COPY workflow_api.py ${COMFYUI_ROOT}/workflow_api.py
# COPY run_workflow.py ${COMFYUI_ROOT}/run_workflow.py

# Jupyter runtime
WORKDIR /workspace

COPY entrypoint /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8888 8188
ENTRYPOINT ["/entrypoint.sh"]
