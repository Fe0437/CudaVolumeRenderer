FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        cmake \
        g++ \
        make \
        git \
        libboost-all-dev \
        libtbb-dev \
        zlib1g-dev \
        libblosc-dev \
        libopenexr-dev \
        libopenvdb-dev \
        python3-openvdb && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
        numpy \
        itk

# Set up working directory
WORKDIR /

# Keep container running
CMD ["tail", "-f", "/dev/null"] 