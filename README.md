# Visual Product Search Engine

Query-by-image fashion retrieval using fine-tuned CLIP, BLIP-2 captioning, and HNSW indexing — built on the [DeepFashion In-Shop](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) benchmark.

---

## How it works

A four-stage offline pipeline indexes the gallery once; queries are served in under a second.

```
Query image → YOLOv8 crop → CLIP encode → HNSW search → BLIP-2 ITM rerank → Results
```

**Offline (gallery):** YOLOv8 localizes each garment → BLIP-2 generates a structured caption → CLIP visual + text embeddings are fused (`α·v + (1−α)·t`) → stored in an HNSW index.

**Online (query):** Crop → CLIP encode → retrieve 2K candidates → rerank by BLIP-2 ITM score → return top K.

---

## Results

Best config: fine-tuned CLIP + BLIP-2, α = 0.7 (mean ± std over 4 seeds).

| Config | R@5 | R@10 | NDCG@10 | mAP@10 |
|---|---|---|---|---|
| A — Vision-only, frozen | 0.39 | 0.44 | 0.42 | 0.29 |
| B — Frozen CLIP + BLIP-2, α=0.7 | 0.45 | 0.47 | 0.45 | 0.30 |
| **C — Fine-tuned + BLIP-2, α=0.7** | **0.66** | **0.74** | **0.90** | **0.54** |

---

## Setup

```bash
pip install open-clip-torch hnswlib ultralytics transformers streamlit
```

Checkpoints and index files are not included. Run the notebooks in order:

| Notebook | Description |
|---|---|
| [dataset-analysis](https://www.kaggle.com/code/mihirkagalkar/dataset-analysis) | Explore and validate the benchmark |
| [clip-finetuning](https://www.kaggle.com/code/pravinkumarv45510/clip-finetuning) | Fine-tune CLIP on DeepFashion pairs |
| [offline-indexing](https://www.kaggle.com/code/mihirkagalkar/offline-indexing) | Build BLIP-2 captions + HNSW index |
| [online-retrieval](https://www.kaggle.com/code/pravinkumarv45510/online-retrieval) | Evaluate retrieval metrics |
| [ablation-study](https://www.kaggle.com/code/adityarao061/ablation-study) | Reproduce ablation table and curves |
| [streamlit-demo](https://www.kaggle.com/code/adityarao061/streamlit-demo) | Interactive demo notebook |

---

## Demo

Run the streamlit-demo notebook, and open the App URL as given by NgrokTunnel. Upload any clothing image. Confirm or reject the YOLO crop, then hit **Search** to retrieve the top-10 gallery matches with ITM and ANN scores.

---
