FROM ubuntu:16.04
LABEL maintainer wilburliu <1293874692@qq.com>

RUN apt-get update && apt-get install -y python-dev python-pip python3-dev python3-pip apache2 libapache2-mod-wsgi-py3 && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install flask -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install redis==3.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install requests==2.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install -U pip && pip3 install numpy==1.18.1 

ENV ROOT=/opt
WORKDIR $ROOT

# Copy over the apache configuration file and enable the site
COPY ./000-default.conf /etc/apache2/sites-available/000-default.conf
RUN a2ensite 000-default
RUN a2enmod headers

CMD ["/usr/sbin/apache2ctl","-DFOREGROUND"]

