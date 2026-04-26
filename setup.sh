pip install uv
uv venv /root/cs336_venv
ln -s /root/cs336_venv .venv
source .venv/bin/activate
uv pip install torch
uv pip install -e ./cs336-basics
uv pip install fire numpy pandas einops jaxtyping pyyaml


# nsys
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb

dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update

apt-cache search nsight-systems

apt-get install -y cuda-nsight-systems-13-2

# Find the actual location of the binary
realpath $(which nsys)

mkdir -p /workspace/tools

cp -a /opt/nvidia/nsight-systems/2025.6.3 /workspace/tools/

# Add the new path to your .bashrc
echo 'export PATH=/workspace/tools/2025.6.3/bin:$PATH' >> ~/.bashrc

# Update the library path so nsys can find its internal components
echo 'export LD_LIBRARY_PATH=/workspace/tools/2025.6.3/target-linux-x64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Apply the changes to your current session
source ~/.bashrc
