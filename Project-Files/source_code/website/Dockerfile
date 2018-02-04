FROM tiangolo/uwsgi-nginx-flask:python2.7

# If STATIC_INDEX is 1, serve / with /static/index.html directly (or the static URL configured)
#ENV STATIC_INDEX 1
ENV STATIC_INDEX 0
ENV STATIC_PATH /app/templates
#ENV UWSGI_READ_TIMEOUT 1200 

#WORKDIR /app

# Add demo app
#COPY . .
COPY timeout.conf ../etc/nginx/conf.d/
COPY ./app /app
WORKDIR .

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python2.7 \
        python2.7-dev \
        rsync \
	vim \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
#        Pillow \
        h5py \
#        ipykernel \
#        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
	nltk 
#        && \
#    python -m ipykernel.kernelspec

#RUN \
#   pip install requirements.txt

#ARG TENSORFLOW_VERSION==1.4.0
#ARG TENSORFLOW_DEVICE==cpu
#ARG TENSORFLOW_APPEND=_gpu
RUN pip --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp27-none-linux_x86_64.whl

#ARG KERAS_VERSION==2.0.8
#ENV KERAS_BACKEND=tensorflow
RUN pip --no-cache-dir install git+https://github.com/fchollet/keras.git@2.0.8

WORKDIR /app

# Make port 80 available to the world outside this container
#EXPOSE 80

# Run main.py when the container launches
#CMD ["python", "main.py"]


