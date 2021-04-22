
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer wilburliu <1293874692@qq.com>



ENV OPENCV_ROOT=/opt/opencv
WORKDIR $OPENCV_ROOT


ENV ROOT=/opt

WORKDIR $ROOT


CMD ["/opt/torch-rest-api/run_torch_rest_api.sh"]
