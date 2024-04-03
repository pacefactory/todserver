#!/bin/bash

# -------------------------------------------------------------------------
# Set image-specific variables

# Set (base) image name
image_name_cpu="todserver_image_cpu"
image_name_gpu="todserver_image_gpu"

dockerfile_name_cpu="Dockerfile_cpu"
dockerfile_name_gpu="Dockerfile_gpu"


# -------------------------------------------------------------------------
# Set CPU vs GPU version

# Prompt the user for input
read -p "Use GPU: y/[n] " user_resp
user_resp_lowcase=${user_resp,,}

if [[ "$user_resp_lowcase" == *'y'* ]]; then
    	echo "Building GPU version..."
	image_name=${image_name_gpu}
    	dockerfile_name=${dockerfile_name_gpu}
else
	echo "Building CPU version..."
	image_name=${image_name_cpu}
    	dockerfile_name=${dockerfile_name_cpu}
fi


# -------------------------------------------------------------------------
# Figure out pathing

# Get shared pathing info
this_script_relative_path=$0
this_script_full_path=$(realpath $this_script_relative_path)
docker_folder_path=$(dirname $this_script_full_path)
build_folder_path=$(dirname $docker_folder_path)

# Get important paths
root_project_folder_path=$(dirname $build_folder_path)
build_name=$(basename $root_project_folder_path)
dockerfile_path="$docker_folder_path/$dockerfile_name"


# -------------------------------------------------------------------------
# Build new image

# Some feedback
echo ""
echo "*** $build_name ***"
echo "Building from dockerfile:"
echo "@ $dockerfile_path"
echo ""
echo ""

# Actual build command
docker build -t $image_name -f $dockerfile_path $root_project_folder_path "$@"

