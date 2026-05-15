# %% [code]
# %% [code]
import os, pickle, json
import numpy as np
import streamlit as st
import torch
from PIL import Image
import hnswlib
import open_clip
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration

st.set_page_config(
    page_title="Product Search Engine",
\    layout="wide"
)

INDEX_DIR  = "./index"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K      = 10

with open(os.path.join(INDEX_DIR, "config.json")) as f:
    cfg = json.load(f)
ALPHA     = cfg["alpha"]
EMBED_DIM = cfg["embed_dim"]

@st.cache_resource
def load_resources():
    # CLIP
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    ckpt_path = "checkpoints/clip_finetuned_full.pt"
    if os.path.exists(ckpt_path):
        clip_model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    clip_model = clip_model.to(DEVICE).eval()

    # YOLO
    yolo = YOLO(cfg.get("yolo_model", "yolov8n.pt"))

    # BLIP-2
    blip_proc  = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE).eval()

    # HNSW index
    hnsw = hnswlib.Index(space="cosine", dim=EMBED_DIM)
    hnsw.load_index(
        os.path.join(INDEX_DIR, "hnsw_gallery.bin"),
        max_elements=cfg["n_gallery"])
    hnsw.set_ef(200)

    # Metadata
    with open(os.path.join(INDEX_DIR, "gallery_meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    return clip_model, preprocess, yolo, blip_proc, blip_model, hnsw, meta


clip_model, preprocess, yolo, blip_proc, blip_model, hnsw, meta = load_resources()


def yolo_crop(image):
    results = yolo(image, verbose=False)
    boxes   = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        b = boxes.xyxy[int(boxes.conf.argmax())].int().tolist()
        return image.crop(b), b
    return image, None


@torch.no_grad()
def encode(img):
    t   = preprocess(img).unsqueeze(0).to(DEVICE)
    emb = clip_model.encode_image(t)
    return (emb / emb.norm(dim=-1, keepdim=True)).cpu().float().numpy()


@torch.no_grad()
def itm_score(img, caption):
    inp = blip_proc(
        images=img, text=caption, return_tensors="pt"
    ).to(DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
    try:
        out    = blip_model(**inp)
        logits = out.itm_score
        return torch.softmax(logits, -1)[0, 1].item()
    except Exception:
        out = blip_model(**inp, labels=inp["input_ids"])
        return -out.loss.item()


st.title("🛍️ Visual Product Search Engine")
st.markdown("""
Upload a clothing image to find visually and semantically similar products.
""")

uploaded = st.file_uploader("Upload a product image", type=["jpg","jpeg","png","webp"])

if uploaded is not None:
    raw_img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(raw_img, caption="Uploaded image", use_column_width=True)

    # Step 1: YOLO crop
    with st.spinner("Detecting product region (YOLO)..."):
        cropped, bbox = yolo_crop(raw_img)

    with col2:
        if bbox:
            st.success(f"Product detected: bbox {bbox}")
        else:
            st.warning("No detection — using full image.")
        st.image(cropped, caption="Cropped product", use_column_width=True)

    # User confirmation
    confirm = st.radio("Use this crop for search?",
                        ["Yes", "No"])

    if st.button("Search"):
        search_img = cropped if confirm.startswith("Y") else raw_img

        with st.spinner("Encoding query with CLIP"):
            q_emb = encode(search_img)

        with st.spinner("Searching index using ANN"):
            labels, distances = hnsw.knn_query(q_emb, k=TOP_K * 2)

        with st.spinner("Re-ranking with BLIP-2 ITM..."):
            scored = []
            for lid, dist in zip(labels[0], distances[0]):
                cap   = meta["captions"][lid]
                score = itm_score(search_img, cap)
                scored.append({
                    "idx": lid,
                    "item_id": meta["item_ids"][lid],
                    "caption": cap,
                    "ann_dist": float(dist),
                    "itm_score": score
                })
            scored.sort(key=lambda x: x["itm_score"], reverse=True)
            results = scored[:TOP_K]

        st.subheader(f"Top {TOP_K} Results")
        cols = st.columns(5)
        for i, res in enumerate(results):
            with cols[i % 5]:
                try:
                    import pandas as pd
                    gallery_df = pd.read_csv("data/gallery.csv")
                    img_path   = gallery_df.iloc[res["idx"]]["full_path"]
                    st.image(Image.open(img_path).convert("RGB"),
                             use_column_width=True)
                except Exception:
                    st.write("[image unavailable]")
                st.caption(
                    f"Rank {i+1}\n"
                    f"ITM: {res['itm_score']:.3f}\n"
                    f"ANN: {res['ann_dist']:.3f}")