export DOCKER_BUILDKIT=1

# ensure that you have generated a ssh key pair
#
# $ ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# $ eval "$(ssh-agent -s)"
# $ ssh-add ~/.ssh/id_rsa
#

if [ $# -lt 2 ]; then
    echo "Specify dockerfile and tag Usage: ./build.sh <dockerfile_path> <imagename:tag> [progress=plain/auto]"
    echo "example: ./build.sh dockerfile.env docker.io/gc625kodifly/llm-image:env"
fi

DOCKERFILE=$1
IMAGE_NAME=$2
PROGRESS_TYPE=$3


if [ -z "$PROGRESS_TYPE" ]; then
    PROGRESS_TYPE="auto"
fi



docker build --progress=${PROGRESS_TYPE} \
--secret id=host_ssh_key,src=/home/${USER}/.ssh/id_rsa \
-f ${DOCKERFILE} \
-t ${IMAGE_NAME} . 