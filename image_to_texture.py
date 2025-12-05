"""
Image to 3D mesh + texture pipeline with BiRefNet background removal.

This script performs the following steps:
1.  **Object Detection (Optional)**: If a text prompt is provided, loads Qwen3-VL to detect objects. All detected objects are cropped and saved to `cropped_images`.
2.  **Background Removal**: Loads BiRefNet to remove the background from images in `cropped_images` (or the original image if no prompt). Results are saved to `bg_removed`.
3.  **Shape Generation**: Loads Hunyuan3D DiT model to generate 3D mesh shapes for each image in `bg_removed`.
4.  **Mesh Processing**: Cleans up the generated meshes.
5.  **Texture Generation (Optional)**: If enabled, loads Hunyuan3D Paint model to generate texture for the meshes.
6.  **Output**: Saves the final meshes (textured or untextured) as GLB files in the run-specific output directory.

VRAM Optimization:
- Models are loaded only when needed and unloaded immediately after use.
- `gc.collect()` and `torch.cuda.empty_cache()` are called to free up memory.
"""

import sys
import argparse
import gc
import torch
import ast
import os
import cv2
import shutil
import datetime
import gradio as gr
from PIL import Image
from torchvision import transforms

sys.path.append('BiRefNet')

from image_proc import refine_foreground
from models.birefnet import BiRefNet
from utils import check_state_dict

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline, 
    FaceReducer, 
    FloaterRemover, 
    DegenerateFaceRemover, 
    MeshlibCleaner
)
from hy3dgen.texgen import Hunyuan3DPaintPipeline

# Global variables for models to allow loading/unloading
qwen_model = None
qwen_processor = None
birefnet = None
pipeline_shape = None
pipeline_texture = None

def parse_args():
    parser = argparse.ArgumentParser(description="Image to 3D mesh + texture pipeline")
    parser.add_argument('--birefnet_device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for BiRefNet')
    parser.add_argument('--qwen_device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for Qwen3-VL')
    return parser.parse_args()

args = parse_args()

num_inference_steps = 50
seed = 42

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()

def load_qwen(device):
    global qwen_model, qwen_processor
    print(f"Loading Qwen3-VL on {device}...")
    model_id = 'Qwen/Qwen3-VL-2B-Instruct'
    qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device
    )
    qwen_processor = AutoProcessor.from_pretrained(model_id)

def unload_qwen():
    global qwen_model, qwen_processor
    print("Unloading Qwen3-VL...")
    del qwen_model
    del qwen_processor
    qwen_model = None
    qwen_processor = None
    cleanup_memory()

def load_birefnet(device):
    global birefnet
    print(f"Loading BiRefNet on {device}...")
    model_name = 'BiRefNet'
    birefnet = BiRefNet(bb_pretrained=False)
    state_dict = torch.load(
        'birefnet_weights/BiRefNet-general-epoch_244.pth', 
        map_location=device
    )
    state_dict = check_state_dict(state_dict)
    birefnet.load_state_dict(state_dict)
    if device == 'cuda':
        torch.set_float32_matmul_precision(['high', 'highest'][0])
    birefnet.to(device)
    birefnet.eval()
    if device == 'cuda':
        birefnet.half()
    print('BiRefNet is ready to use.')

def unload_birefnet():
    global birefnet
    print("Unloading BiRefNet...")
    del birefnet
    birefnet = None
    cleanup_memory()

def load_hunyuan_shape():
    global pipeline_shape
    print("Loading Hunyuan3D Shape model on cuda...")
    pipeline_shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mini',
        subfolder='hunyuan3d-dit-v2-mini',
        use_safetensors=True,
        device='cuda'
    )

def unload_hunyuan_shape():
    global pipeline_shape
    print("Unloading Hunyuan3D Shape model...")
    del pipeline_shape
    pipeline_shape = None
    cleanup_memory()

def load_hunyuan_texture():
    global pipeline_texture
    print("Loading Hunyuan3D Texture model on cuda...")
    pipeline_texture = Hunyuan3DPaintPipeline.from_pretrained(
        'tencent/Hunyuan3D-2',
        device='cuda',
    )

def unload_hunyuan_texture():
    global pipeline_texture
    print("Unloading Hunyuan3D Texture model...")
    del pipeline_texture
    pipeline_texture = None
    cleanup_memory()

# BiRefNet image transforms.
def get_transform_image(model_name='BiRefNet'):
    return transforms.Compose([
        transforms.Resize((1024, 1024) if '_HR' not in model_name else (2048, 2048)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def qwen_object_boxes(model, processor, image_path, prompt):
    """Use Qwen3-VL to generate bounding boxes for natural-language prompts."""
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': image_path},
            {'type': 'text', 'text': prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt'
    ).to(model.device)

    generated = model.generate(**inputs, max_new_tokens=4096)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    # Parse Qwen output.
    json_str = decoded[8:-3] if decoded.startswith('```json') else decoded
    detections = ast.literal_eval(json_str)
    return detections

def crop_dets(image_path, detections, save_dir):
    """
    Crop the detection area of objects and save them.
    """
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = image_bgr.shape

    print(f"Detections: {detections}")
        
    count = 0
    for i, det in enumerate(detections):
        box = det['bbox_2d']
        x1 = int(box[0] / 1000 * w)
        y1 = int(box[1] / 1000 * h)
        x2 = int(box[2] / 1000 * w)
        y2 = int(box[3] / 1000 * h)

        crop = image_rgb[y1:y2, x1:x2]
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f'crop_{i}.png'), crop_bgr)
        count += 1
    
    return count

def remove_bg(image_path, device):
    """Feed image to BiRefNet for background removal."""
    # Assumes BiRefNet is already loaded
    
    image = Image.open(image_path)
    transform_image = get_transform_image()
    input_images = transform_image(image).unsqueeze(0).to(device)
    if device == 'cuda':
        input_images = input_images.half()
    
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    pred_pil = pred_pil.resize(image.size)
    image_masked = refine_foreground(image, pred_pil)
    image_masked.putalpha(pred_pil)
    
    return image_masked

def setup_directories():
    """Create timestamped directories for this run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_outdir = os.path.join('outputs', timestamp)
    crop_dir = os.path.join('cropped_images', timestamp)
    bg_dir = os.path.join('bg_removed', timestamp)
    
    for d in [run_outdir, crop_dir, bg_dir]:
        os.makedirs(d, exist_ok=True)
        
    return run_outdir, crop_dir, bg_dir

def image_to_3d(text, image_path, do_texture):
    run_outdir, crop_dir, bg_dir = setup_directories()
    
    fix_holes = False
    
    # Object Detection & Cropping
    images_to_process = []
    
    if len(text) > 0:
        load_qwen(args.qwen_device)
        prompt = f"Locate every instance that belongs to the following categories: {text}. Report bbox coordinates in JSON format."
    
        detections = qwen_object_boxes(qwen_model, qwen_processor, image_path, prompt)
        print(f"Qwen3-VL detections: {len(detections)} objects")
        unload_qwen()
        
        num_crops = crop_dets(image_path, detections, crop_dir)
        if num_crops > 0:
            for f in sorted(os.listdir(crop_dir)):
                images_to_process.append(os.path.join(crop_dir, f))
    
    # If no prompt or no detections, use original image
    if not images_to_process:
        images_to_process.append(image_path)

    # Background Removal
    load_birefnet(args.birefnet_device)
    processed_images = []
    for i, img_path in enumerate(images_to_process):
        image_masked = remove_bg(img_path, args.birefnet_device)
        save_path = os.path.join(bg_dir, f'bg_removed_{i}.png')
        image_masked.save(save_path)
        processed_images.append(save_path)
    unload_birefnet()

    # Shape Generation
    load_hunyuan_shape()
    meshes = []
    for img_path in processed_images:
        mesh = pipeline_shape(
            image=img_path,
            num_inference_steps=num_inference_steps,
            generator=torch.manual_seed(seed)
        )[0]
        meshes.append(mesh)
    unload_hunyuan_shape()

    # 4. Mesh Processing & Texture Generation
    final_paths = []
    
    if do_texture:
        load_hunyuan_texture()

    for i, mesh in enumerate(meshes):
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        if fix_holes:
            mesh = MeshlibCleaner()(mesh)
        mesh = FaceReducer()(mesh)
        
        if do_texture:
            mesh = pipeline_texture(mesh, Image.open(processed_images[i]))
        
        save_path = os.path.join(run_outdir, f'model_{i}.glb')
        mesh.export(save_path)
        final_paths.append(save_path)

    if do_texture:
        unload_hunyuan_texture()

    # Pad with None to match fixed output count (8)
    while len(final_paths) < 8:
        final_paths.append(None)
        
    return final_paths[:8]

with gr.Blocks() as demo:
    gr.Markdown("# Image to 3D Mesh + Texture Pipeline")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type='filepath', label="Input Image")
            prompt = gr.Text(label="Object Prompt (Optional, e.g., 'cup, spoon')")
            do_texture = gr.Checkbox(label="Generate Texture", value=True)
            submit_btn = gr.Button("Generate 3D Models")
        
        with gr.Column(scale=3):
            with gr.Row():
                out1 = gr.Model3D(label="Result 1", height=600)
                out2 = gr.Model3D(label="Result 2", height=600)
            with gr.Row():
                out3 = gr.Model3D(label="Result 3", height=600)
                out4 = gr.Model3D(label="Result 4", height=600)
            with gr.Row():
                out5 = gr.Model3D(label="Result 5", height=600)
                out6 = gr.Model3D(label="Result 6", height=600)
            with gr.Row():
                out7 = gr.Model3D(label="Result 7", height=600)
                out8 = gr.Model3D(label="Result 8", height=600)

    submit_btn.click(
        fn=image_to_3d,
        inputs=[prompt, input_image, do_texture],
        outputs=[out1, out2, out3, out4, out5, out6, out7, out8]
    )

if __name__ == "__main__":
    demo.launch(share=True)