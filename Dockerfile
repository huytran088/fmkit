# Start from an official Python 3.11 image.
FROM python:3.11-slim

# Set a working directory inside the container
WORKDIR /app

# Install system packages
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y p7zip \
    p7zip-full \
    zip \
    unzip \
    xz-utils \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages compatible with Python 3.11
RUN pip install --no-cache-dir \
    numpy>=1.24,<3 \
    matplotlib>=3.7 \
    pandas>=2.0 \
    scikit-learn>=1.3 \
    scipy>=1.11 \
    jupyter

EXPOSE 8888

# Copy your project code into the container
# (You can comment this out if you just want a shell)
COPY . .

# By default, when the container runs, open a bash shell
CMD ["bash"]
