FROM python:3

FROM python:3
MAINTAINER  Siddharth Yadav "siddharth16268@iiitd.ac.in"

# If needed, install system dependencies here

# Starts installing python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Copying code to image OR use -v "$PWD":/app/
# ADD . /app

WORKDIR /app


CMD ["bash"]
