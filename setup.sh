# Create build_setup dir in root project directory and clone Hunyuan3D-2 inside that.
mkdir build_setup
cd build_setup
# git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
# git clone https://github.com/jclarkk/Hunyuan3D-2.git
git clone https://github.com/sovit-123/Hunyuan3D-2.git
cd Hunyuan3D-2

echo "Patching Nvdiffrast library. Pinning requirement to v0.3.4..."
sed -i "s|git+https://github.com/NVlabs/nvdiffrast.git|git+https://github.com/NVlabs/nvdiffrast.git@v0.3.4|" requirements.txt

pip install -r requirements.txt
pip install -e .
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
# use the following only if using jclarkk/Hunyuan3D-2.git
pip install -e .

cd ../../../../../

# BiRefNet cloned in project root directory.
git clone https://github.com/ZhengPeng7/BiRefNet.git
cd BiRefNet
pip install -r requirements.txt

cd ../
pip install -r hunyuan3d_final_req.txt