#!/bin/bash

# -------------------------------------------------------------------------
# Set image-specific variables

# Set naming
image_name_cpu="todserver_image_cpu"
image_name_gpu="todserver_image_gpu"
container_name="todserver"

# Set networking
network_setting="host"

# Set volume pathing
volume_name="local_todserver_volume"
container_volume_path="/home/storage"


# -------------------------------------------------------------------------
# Set CPU vs GPU version

# Prompt the user for input
read -p "Use GPU: y/[n] " user_resp
user_resp_lowcase=${user_resp,,}

gpu_arg=""
if [[ "$user_resp_lowcase" == *'y'* ]]; then
    	echo "Running GPU version..."
	image_name=${image_name_gpu}
	gpu_arg="--gpus all"
else
	echo "Running CPU version..."
	image_name=${image_name_cpu}
fi


# -------------------------------------------------------------------------
# Automated commands

# Some feedback while stopping the container
echo ""
echo "Stopping existing container..."
docker stop $container_name > /dev/null 2>&1
echo "  --> Success!"

# Some feedback while removing the existing container
echo ""
echo "Removing existing container..."
docker rm $container_name > /dev/null 2>&1
echo "  --> Success!"

echo ""
echo "Creating named volume ($volume_name)"
echo "(to delete the volume, use: docker volume rm $volume_name)"
docker volume create $volume_name > /dev/null 2>&1
echo "  --> Success!"

# Now run the container
echo ""
echo "Running container ($container_name)"
docker run -d \
	   $gpu_arg \
           --network=$network_setting \
           --mount source=$volume_name,target=$container_volume_path \
           --name $container_name \
           $image_name \
           > /dev/null
echo "  --> Success!"


# Some final feedback
echo ""
echo "-----------------------------------------------------------------"
echo ""
echo "To check the status of all running containers use:"
echo "docker ps -a"
echo ""
echo "To stop this container use:"
echo "docker stop $container_name"
echo ""
echo "To 'enter' into the container (for debugging/inspection) use:"
echo "docker exec -it $container_name bash"
echo ""
echo "-----------------------------------------------------------------"
echo ""
