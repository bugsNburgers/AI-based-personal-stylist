
import os
import numpy as np
import torch
import clip
from PIL import Image


# CONFIG

OUTPUTS_DIR = "outputs"
DEVICE = "cpu"   # CPU only (explicit)

# LOAD CLIP MODEL
print("[INFO] Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()
print("[INFO] CLIP loaded on CPU")

def embed_image(img_path):
    # Garment crops are saved as transparent PNGs (RGBA). CLIP expects RGB.
    # If we convert RGBA->RGB directly, transparency is typically filled with black,
    # which re-introduces a background bias. Composite onto a neutral background first.
    image_rgba = Image.open(img_path).convert("RGBA")
    white_bg = Image.new("RGBA", image_rgba.size, (255, 255, 255, 255))
    image_rgb = Image.alpha_composite(white_bg, image_rgba).convert("RGB")

    image = preprocess(image_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy()[0]

if __name__ == "__main__":
    for image_id in os.listdir(OUTPUTS_DIR):
        image_folder = os.path.join(OUTPUTS_DIR, image_id)

        if not os.path.isdir(image_folder):
            continue

        for file in os.listdir(image_folder):
            if not file.endswith(".png"):
                continue

            img_path = os.path.join(image_folder, file)
            emb_path = img_path.replace(".png", ".npy")

            if os.path.exists(emb_path):
                continue  # avoid recompute

            print(f"[EMBED] {img_path}")
            emb = embed_image(img_path)
            np.save(emb_path, emb)

    print("[DONE] All embeddings extracted")
