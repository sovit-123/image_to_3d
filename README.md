# Image to 3D Mesh and Inpainting/Texturing Pipeline (with Runpod Docker Image)

This project aims to make the image-to-3D mesh and inpainting pipeline a seamless experience.

***Powered by Hunyuan3D 2.0, Qwen3-VL, and BiRefNet models.***

## To Run Locally

```
git clone https://github.com/sovit-123/image_to_3d.git
cd image_to_3d
```

Create a `birefnet_weights` directory and download and put [these BiRefNet weights](https://drive.google.com/file/d/1_IfUnu8Fpfn-nerB89FzdNXQ7zk6FKxc/view) in the directory.

Create a Python 3.10 environment, activate it, and run setup.

```
sh setup.sh
```

After setup completes, run `image_to_texture.py`.

```
python image_to_texture.py
```

## Run on Runpod via Docker Image

Directly run on Runpod without the hassle of local setup and GPU constraint.

Launch any pod with >= 20GB VRAM with [this Runpod template](https://console.runpod.io/deploy?template=kj1pcha6vo&ref=9c1nq0qt).

After the pod starts:

You can SSH into the pod or open via Jupyter Lab.

Open a terminal and execute:

1. `source venv/bin/activate` => The virtual environment is already create via Dockerfile in the `/workspace` directory. You just need to enable.
2. `sh run.sh` => This will complete the setup and launch the Gradio application as well.

You should see a public Gradio link. Open it and start playing around.
