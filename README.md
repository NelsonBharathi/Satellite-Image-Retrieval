# A Cross-Modal Retrieval and Change Detection Framework for Location-Specific Remote Sensing Data

This repository presents an AI-driven **multimodal remote-sensing framework** that enables **natural-language–based retrieval of satellite imagery** and performs **semantic land-cover segmentation and temporal change detection** for location-specific analysis.

The system is evaluated on multi-year satellite imagery collected from Chennai taluks (2014–2024) and integrates state-of-the-art NLP and computer vision models to bridge the semantic gap between textual queries and visual remote-sensing data.

---

## Abstract

Traditional remote-sensing platforms rely heavily on structured metadata and manual filtering, limiting accessibility for non-expert users. This project introduces a **cross-modal retrieval framework** that allows users to query satellite imagery using **natural language**, retrieve semantically relevant images, and perform **land-cover segmentation** and **pixel-level change detection** across years.

The framework combines **transformer-based intent detection**, **vision–language embeddings**, **vector similarity search**, and **deep semantic segmentation**, producing interpretable visual outputs and quantitative change statistics.

---

## Key Contributions

- Natural-language–based satellite image retrieval  
- Cross-modal semantic alignment of text and imagery  
- Automated land-cover segmentation (vegetation, water, urban)  
- Year-wise pixel-level change detection and analysis  
- Scalable and interpretable remote-sensing workflow  

---

## Dataset Description

- **Geographic Region:** Chennai Taluks  
- **Time Span:** 2014 – 2024  
- **Total Images:** 2,304  
- **Image Resolution:** 512 × 512 pixels  
- **Spatial Resolution:** ~30 meters per pixel  
- **Land-Cover Classes:** Vegetation, Water, Urban  

Satellite imagery was collected using multi-year observations from publicly available sources.

---

## Models and Techniques Used

### Transformer-Based Intent Detection
A fine-tuned lightweight transformer model is used to classify user queries into supported intent categories, enabling accurate query interpretation with low inference latency.

### Vision–Language Embeddings
A joint text–image embedding model aligns satellite images and natural-language queries into a shared semantic space, enabling flexible semantic retrieval beyond keyword matching.

### Vector Similarity Search
A high-performance vector index enables efficient Top-K similarity search over large-scale image embeddings using cosine similarity.

### Semantic Segmentation
A deep encoder–decoder architecture with multi-scale context aggregation is employed to segment land-cover classes from satellite imagery.

### Temporal Change Detection
Pixel-level differencing across segmentation masks enables quantification of land-cover gain, loss, and net change over time.

---

## Project Structure (Recommended)

```text
.
├── app.py
├── requirements.txt
├── data/
│   ├── 2014/
│   ├── 2015/
│   └── ...
├── metadata/
│   └── metadata.csv
├── models/
│   ├── intent_classifier/
│   └── segmentation_model/
├── index/
│   └── vector.index
├── scripts/
│   ├── build_embeddings.py
│   ├── build_index.py
│   ├── segmentation.py
│   └── change_detection.py
└── README.md

