FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm_20250321

ARG USE_LOCAL_WHEEL=false

# Install system dependencies
RUN apt-get update && apt-get install -y curl gnupg

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Add the Cloud Storage FUSE distribution URL as a package source
RUN echo "deb https://packages.cloud.google.com/apt gcsfuse-bullseye main" | tee /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install the Google Cloud SDK and GCS fuse
RUN apt-get update && apt-get install -y google-cloud-sdk git fuse gcsfuse && gcsfuse -v

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1

WORKDIR /workspaces

RUN pip install jax==0.5.4.dev20250321 jaxlib==0.5.4.dev20250321 \
    -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html

COPY . /workspaces/diffusers/

WORKDIR /workspaces/diffusers/examples/research_projects/pytorch_xla/training/text_to_image/
RUN pip3 install -r requirements_sdxl.txt
RUN pip3 install pillow --upgrade

WORKDIR /workspaces/diffusers/
RUN pip3 install .