FROM continuumio/miniconda3

ENV FTP_PROXY=http://proxy.swmed.edu:3128/
ENV HTTPS_PROXY=http://proxy.swmed.edu:3128/
ENV HTTP_PROXY=http://proxy.swmed.edu:3128/
ENV ftp_proxy=http://proxy.swmed.edu:3128/
ENV http_proxy=http://proxy.swmed.edu:3128/
ENV https_proxy=http://proxy.swmed.edu:3128/

## Make RUN commands use the new environment:
COPY environment.yaml /
RUN apt-get update && apt-get install -yy build-essential && apt-get clean && conda env create -f /environment.yaml

## Downloads scripts
COPY hd_wsi.v2 /usr/src/hd_wsi/
# RUN git clone https://github.com/impromptuRong/hd_wsi /usr/src/hd_wsi

## Download models
RUN pip install gdown
RUN mkdir -p /usr/src/hd_wsi/selected_models/benchmark_lung/
RUN gdown 131RQwmrQeonwuLr46L06gWZ8Jv60opSt -O /usr/src/hd_wsi/selected_models/benchmark_lung/
RUN mkdir -p /usr/src/hd_wsi/selected_models/benchmark_nucls_paper/
RUN gdown 131zR4g-V1wmjXBhmzuEGnqt-ttzNuSPK -O /usr/src/hd_wsi/selected_models/benchmark_lung/

WORKDIR /usr/src/hd_wsi
RUN ls
# uvicorn app:app --host 0.0.0.0 --port 5001 --workers 32 --log-level debug --reload
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ml_env", \
            "uvicorn", "app:app", "--host", "0.0.0.0:", "--port", "5000", 
            "--workers", "16", "--log-level", "debug", "--reload"]

# ENTRYPOINT . "/opt/conda/etc/profile.d/conda.sh" && export PATH="/opt/conda/bin:$PATH" && conda activate ml_env0 && python /usr/src/hd_wsi/deepzoom_multiserver.py --data_path /slides_folder -l 0.0.0.0

# -v `readlink -f lung_best.float16.torchscript.pt`:/usr/src/hd_wsi/selected_models/benchmark_lung/lung_best.float16.torchscript.pt
# /usr/src/hd_wsi/selected_models/benchmark_lung/

# SHELL ["conda", "run", "-n", "ml_env", "/bin/bash", "-c"]

# ## Download sample slides
# RUN mkdir -p /usr/src/slides_folder
# RUN gdown 1mgN0ja5aoBIPtvEvJix3hBsOid3gISl8 -O /usr/src/slides_folder/
