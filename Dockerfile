# Start from an official Python 3.6.9 image.
# This image is based on an older Debian (Buster),
# where 3.6.9 and its packages work perfectly.
FROM python:3.6.9-slim-buster

# Set a working directory inside the container
WORKDIR /app

# Install your packages
# This command first points apt to the EOL archive,
# then updates and installs vim.
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i '/buster-updates/d' /etc/apt/sources.list && \
    apt-get update  -y && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get -y autoremove && \
    apt-get clean
RUN apt-get install -y p7zip \
    p7zip-full \
    unace \
    zip \
    unzip \
    xz-utils \
    sharutils \
    uudeview \
    mpack \
    arj \
    cabextract \
    file-roller \
    && rm -rf /var/lib/apt/lists/*
RUN pip install numpy==1.19.5 matplotlib==3.1.2 jupyter

EXPOSE 8888

# Copy your project code into the container
# (You can comment this out if you just want a shell)
COPY . .

# By default, when the container runs, open a bash shell
CMD ["bash"]
