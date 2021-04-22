
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer Jin Xiaoping <xiaoping.jin@hanshow.com>

RUN apt-get update && apt-get remove x264 libx264-dev && \
	apt-get install -y --no-install-recommends apt-utils \
        build-essential \
        checkinstall \
        pkg-config \
        yasm \
        cmake \
        vim \
        git \
        gfortran \
        libjpeg8-dev \
        libjasper-dev \
        libpng12-dev \
        libtiff5-dev \
        libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev \
        libxine2-dev libv4l-dev \
        libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev \
        libgtk2.0-dev libtbb-dev \
        libatlas-base-dev \
        libfaac-dev libmp3lame-dev libtheora-dev \
        libvorbis-dev libxvidcore-dev \
        libopencore-amrnb-dev libopencore-amrwb-dev \
        x264 v4l-utils \
        libgphoto2-dev libeigen3-dev libhdf5-dev doxygen \
        python-dev python-pip python3-dev python3-pip \
        wget \
        mysql-client \
        libmysqlclient-dev \
        libatlas-base-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        lxml\
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-numpy \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libboost-all-dev python-skimage && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip numpy && pip3 install -U pip numpy

ENV OPENCV_ROOT=/opt/opencv
WORKDIR $OPENCV_ROOT
RUN git clone https://github.com/opencv/opencv.git . && git checkout 3.4 && \
	mkdir build && cd build && \
	cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_OPENGL=ON \
      -D BUILD_EXAMPLES=ON .. && make -j"$(nproc)" && make install

RUN echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf && ldconfig

RUN apt-get update && \
  apt-get install libmysqlclient-dev -y

RUN pip install -U pip 
RUN pip2 install Pillow \
                azure-storage \
                azure-eventhub \
                sqlalchemy \
                pymysql \
                mysql-connector-python \
                mysqlclient    

RUN pip2 install configparser

ENV ROOT=/opt


RUN pip2 install protobuf


RUN pip2 install easydict && pip2 install cython 
RUN pip2 install pyyaml
RUN apt-get update && apt-get install -y python3-setuptools && rm -rf /var/lib/apt/lists/*
RUN pip3 install easydict && pip3 install cython && pip3 install pyyaml
RUN apt-get update && apt-get install -y python3-skimage && rm -rf /var/lib/apt/lists/*
RUN pip3 install protobuf
RUN apt-get update && apt-get install -y apache2 libapache2-mod-wsgi-py3 && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip 
RUN pip3 install flask

RUN pip3 install redis==3.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install requests==2.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
WORKDIR $ROOT


CMD ["/opt/caffe-rest-api/run_caffe_rest_api.sh"]
