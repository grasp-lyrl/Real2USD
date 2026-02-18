# the point of the dockerfile is the create a new user that is not root
# so that when we build ros packages and new files they dont have permission issues

# check you user id (outside of docker) by running: id -u
# adjust the uid in the dockerfile to match your host user id

# base from ros2 humble ubuntu 22.04
FROM osrf/ros:humble-desktop-full-jammy
# create a user that is not root called 'me' with UID 1000 to match host user
# This ensures mounted volumes have correct permissions (host user typically has UID 1000)
RUN groupadd -g 1000 me 2>/dev/null || true && \
    useradd --create-home --shell /bin/bash --uid 1000 --gid 1000 me
# add new user 'me' to sudo group
RUN usermod -aG sudo me
# sudo password is '123'
RUN echo 'me:123' | chpasswd
# switch to user 'me' with working directory /home/me
USER me
WORKDIR /home/me