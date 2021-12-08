#!/bin/bash
cache_option= #'--no-cache'
arg_option= #"--build-arg KALDI_ROOT"
image_name=pororo
version=$(cat VERSION)
set -e


# dockerize
docker build $cache_option $arg_option -t $image_name:$version --build-arg CACHEBUST=$(date +%s) .

# push image into AWS ECR
ECS_REPO=161969600347.dkr.ecr.ap-northeast-2.amazonaws.com/zeroth/pororo
docker tag $image_name:$version $ECS_REPO:latest
docker tag $image_name:$version $ECS_REPO:$version
login_cmd_on_aws=$(aws ecr get-login --no-include-email)
echo $login_cmd_on_aws | sh -x
docker push $ECS_REPO:latest
docker push $ECS_REPO:$version

#docker run --rm -it \
  #$image_name:$(cat VERSION) /bin/bash
  #-v /home/users/lucasjo/_product_/zeroth_ee/dev_env/:/opt/zeroth_ee/dev_env/ \
