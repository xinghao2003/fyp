# Use Ubuntu as base image
FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TA_LIBRARY_PATH=/usr/lib
ENV TA_INCLUDE_PATH=/usr/include

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    ca-certificates \
    sudo \
    software-properties-common \
    build-essential \
    gcc \
    g++ \
    make \
    pkg-config \
    screen \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Download and install ta-lib C library
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_amd64.deb \
    && sudo dpkg -i ta-lib_0.6.4_amd64.deb \
    && rm ta-lib_0.6.4_amd64.deb \
    && ldconfig || \
    # Fallback: compile from source if .deb fails
    (echo "Package installation failed, compiling from source..." && \
    wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.6.4-src.tar.gz && \
    ldconfig)

# Verify TA-Lib installation
RUN echo "Verifying TA-Lib installation..." && \
    ldconfig -p | grep ta-lib || echo "TA-Lib not found in ldconfig, but may still work" && \
    find /usr -name "*ta_lib*" -o -name "*talib*" 2>/dev/null | head -10 || echo "TA-Lib files check complete"

# Install Python 3.11 specifically with development headers
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    python3.11-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11 using the recommended method
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set Python 3.11 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install VS Code CLI
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz \
    && tar -xf vscode_cli.tar.gz \
    && mv code /usr/local/bin/ \
    && rm vscode_cli.tar.gz

# Clone repository during build using HTTPS
WORKDIR /workspace
RUN git clone https://github.com/xinghao2003/fyp.git

# Setup Python virtual environment and install requirements during build
WORKDIR /workspace/fyp
RUN python3.11 -m venv .venv && \
    . .venv/bin/activate && \
    pip install --upgrade pip && \
    pip install numpy

# Install requirements if they exist
RUN . .venv/bin/activate && \
    if [ -f "requirements.txt" ]; then \
    pip install -r requirements.txt; \
    else \
    echo "No requirements.txt found during build"; \
    fi

# Copy startup script
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

# Set the startup script as entrypoint
ENTRYPOINT ["/startup.sh"]