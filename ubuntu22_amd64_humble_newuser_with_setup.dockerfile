# Example: Hybrid approach - install system deps in Dockerfile, keep ROS2 builds in setup.sh
# This is an OPTIONAL alternative - your current approach is fine!

# the point of the dockerfile is the create a new user that is not root
# so that when we build ros packages and new files they dont have permission issues

# base from ros2 humble ubuntu 22.04
FROM osrf/ros:humble-desktop-full-jammy

# Install system dependencies as root
RUN apt-get update \
    && apt-get -y --quiet --no-install-recommends install \
    gcc \
    git \
    python3 \
    python3-pip \
    wget \
    clang \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge as root (system-wide)
RUN ARCH=$(uname -m) \
    && if [ "$ARCH" = "x86_64" ]; then \
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"; \
    else \
        echo "Unsupported architecture: $ARCH"; exit 1; \
    fi \
    && wget -q "$MINIFORGE_URL" -O /tmp/Miniforge3.sh \
    && bash /tmp/Miniforge3.sh -b -p /opt/miniforge3 \
    && rm /tmp/Miniforge3.sh \
    && /opt/miniforge3/bin/conda config --set auto_activate_base false

# Make miniforge available to all users
RUN echo 'export PATH="/opt/miniforge3/bin:$PATH"' >> /etc/profile.d/miniforge.sh

# create a user that is not root called 'me'
RUN useradd --create-home --shell /bin/bash me
# add new user 'me' to sudo group
RUN usermod -aG sudo me
# sudo password is '123'
RUN echo 'me:123' | chpasswd

# switch to user 'me' with working directory /home/me
USER me
WORKDIR /home/me

# Initialize conda for user 'me'
RUN echo 'if [ -f /opt/miniforge3/etc/profile.d/conda.sh ]; then source /opt/miniforge3/etc/profile.d/conda.sh; fi' >> ~/.bashrc

