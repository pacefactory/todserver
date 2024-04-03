
# This is a helper script, used to automate model downloads
# - This is partly for documentation (record where to get model files)
# - You can alternatively download them manually, and not use this script!

# Set up folder pathing
base_folder_path="storage/models"
gdino_path=$base_folder_path/"gdino"

echo ""
echo "Downloading model files"

echo ""
echo "Downloading GDINO-Tiny"
wget -P $gdino_path --no-clobber \
	https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

echo ""
echo "Downloading GDINO-Base"
wget -P $gdino_path --no-clobber \
	https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

