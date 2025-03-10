FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# build and fill directories.
RUN mkdir /cfg /build &&\
    chmod 755 /cfg &&\
    chmod 777 /build

ADD ./config/* /cfg/.
ADD ./scripts/* /build/.

# installs for environment.
RUN chmod +x /build/* &&\
    /build/install-pkgs.sh

ADD ./late_config/* /cfg/.
ADD ./late_scripts/* /build/.

RUN chmod +x /build/*

ARG DOCKER_USER
ARG UID
ARG GID
ARG GIT_USERNAME
ARG GIT_EMAIL

RUN groupadd -g $GID $DOCKER_USER &&\
    useradd -u $GID -g $GID -s /bin/bash $DOCKER_USER -m&&\
    mkdir  /datasets /home/$DOCKER_USER/.aws &&\
    chown -R $UID:$GID /workspace /datasets

RUN /build/aws-cli.sh
USER $DOCKER_USER
RUN /build/pip-install.sh
RUN /build/vs-code-install.sh $DOCKER_USER
RUN /build/git-config.sh $GIT_USERNAME $GIT_EMAIL

USER root
RUN service ssh start
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]