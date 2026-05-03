# 1. Nuke the ancient Ubuntu package
sudo apt-get purge -y nsight-systems nsight-systems-cli
sudo apt-get autoremove -y

# 2. Hook WSL into NVIDIA's official CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 3. Install the official, fully-loaded CLI
sudo apt-get install -y nsight-systems-cli
