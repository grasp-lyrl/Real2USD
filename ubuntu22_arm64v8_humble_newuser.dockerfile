# the point of the dockerfile is the create a new user that is not root
# so that when we build ros packages and new files they dont have permission issues

# base from ros2 humble ubuntu 22.04
FROM arm64v8/ros:humble-perception-jammy
# create a user that is not root called 'me'
RUN useradd --create-home --shell /bin/bash me
# add new user 'me' to sudo group
RUN usermod -aG sudo me
# sudo password is '123'
RUN echo 'me:123' | chpasswd
# switch to user 'me' with working directory /home/me
USER me
WORKDIR /home/me
