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

## System Architecture (High-Level)

1. **Query Understanding**
   - Intent classification using a fine-tuned transformer model
   - Extraction of theme (vegetation / water / urban), location, and temporal constraints

2. **Cross-Modal Retrieval**
   - Text and image encoding using a shared embedding space
   - Top-K semantic retrieval via vector similarity search

3. **Image Analysis**
   - Semantic segmentation of retrieved satellite images
   - Computation of land-cover distribution statistics

4. **Change Detection**
   - Binary differencing of segmentation masks across years
   - Gain, loss, and net-change computation

5. **Visualization and Summary**
   - Overlay visualizations and interpretable analytical summaries

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

### 1. Transformer-Based Intent Detection
A fine-tuned lightweight transformer model is used to classify user queries into supported intent categories, enabling accurate query interpretation with low inference latency.

### 2. Vision–Language Embeddings
A joint text–image embedding model aligns satellite images and natural-language queries into a shared semantic space, enabling flexible semantic retrieval beyond keyword matching.

### 3. Vector Similarity Search
A high-performance vector index enables efficient Top-K similarity search over large-scale image embeddings using cosine similarity.

### 4. Semantic Segmentation
A deep encoder–decoder architecture with multi-scale context aggregation is employed to segment land-cover classes from satellite imagery.

### 5. Temporal Change Detection
Pixel-level differencing across segmentation masks enables quantification of land-cover gain, loss, and net change over time.

---

## Results Summary

- Efficient semantic retrieval with low query latency  
- Improved retrieval relevance compared to keyword-based baselines  
- Consistent segmentation performance across land-cover classes  
- Interpretable visualization of spatio-temporal changes  

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

