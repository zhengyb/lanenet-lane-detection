# Use the NVIDIA TensorFlow image as the base
FROM nvcr.io/nvidia/tensorflow:23.03-tf1-py3

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file from your build context to /app in the container
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install the dependencies listed in requirements.txt
RUN pip install -r /app/requirements.txt

# Default command is to launch a bash shell
CMD ["bash"]

