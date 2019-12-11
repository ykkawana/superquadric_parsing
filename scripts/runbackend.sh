docker run \
--runtime=nvidia \
--net=host \
--ipc=host \
-e http_proxy=$http_proxy \
-e https_proxy=$https_proxy \
-e no_proxy=$no_proxy \
-e DISPLAY=$DISPLAY \
-e CONTAINER_ID={{.ID}} \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-v $PWD:/workspace \
-v $HOME/.aws:/root/.aws \
-v $HOME/.Xauthority:/root/.Xauthority \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
--rm -ti \
ubuntu18-torch:suqerquadric_parsing
#ubuntu18-torch:latest

#-e NVIDIA_DRIVER_CAPABILITIES=all \
#nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
#172.31.122.2:5000/mltools/cuda-pytorch:latest
#nvidia/cudagl:10.0-devel-ubuntu16.04
