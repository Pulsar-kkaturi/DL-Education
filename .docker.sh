# Docker build & run shell script
data=$1
mode=$2
if [ $mode = "build" ]
then
    echo "Docker Build mode"
    docker build -t dl-study:$USER . 
elif [ $mode = "run" ]
then
    echo "Docker run mode"
    docker run \
    --name dls-$USER \
    --gpus all \
    -v $HOME/.vscode-server:/home/student/.vscode-server \
    -v $PWD:/home/student/Projects \
    -v $data:/home/student/Datasets \
    -it dl-study:$USER
else
    echo "!ERROR! Please insert correct mode! (build or run)"
fi