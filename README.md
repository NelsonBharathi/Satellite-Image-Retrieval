# A Cross-modal Retrieval and Change Detection Framework for Location-Specific Remote Sensing Data

An AI-driven **multimodal remote-sensing system** that lets users retrieve satellite imagery using **natural-language queries** and performs **land-cover segmentation + temporal change detection** for location-specific analysis (Chennai taluks, 2014–2024).  [oai_citation:1‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)

---

## Project Abstract
Traditional remote-sensing platforms often rely on structured metadata and manual filters. This project enables **natural-language-based retrieval** of satellite images and provides **analysis summaries**, including **segmentation overlays** (vegetation / water / urban) and **year-wise change detection** metrics.  [oai_citation:2‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)

---

## Key Features
- **Natural language query interface** (e.g., “Vegetation in Guindy 2020”, “Urban growth Tambaram 2024”)  [oai_citation:3‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)
- **Intent detection** using a fine-tuned **DistilBERT** classifier (visualization / temporal / comparison / descriptive)  [oai_citation:4‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)
- **Cross-modal semantic retrieval** using **CLIP embeddings + FAISS** vector similarity search  [oai_citation:5‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)
- **Semantic segmentation** with **DeepLabV3+** for land-cover extraction (vegetation, water, urban)  [oai_citation:6‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)
- **Pixel-level change detection** (gain / loss / net change) with interpretable change maps  [oai_citation:7‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)
- Interactive visualization layer (report mentions Streamlit-based interface)  [oai_citation:8‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)

---

## System Architecture (High-level)
1. **Query Understanding**
   - DistilBERT intent classification
   - Theme extraction (vegetation / water / urban)
   - Year or year-range extraction
   - Fuzzy location matching (token overlap / fuzzy similarity)
2. **Cross-modal Retrieval**
   - Encode query text → 512-d embedding (CLIP Text Encoder)
   - Retrieve Top-K matching satellite images using **FAISS** (cosine similarity via inner product on normalized vectors)
3. **Image Analysis**
   - Segment retrieved images using **DeepLabV3+**
   - Generate theme masks and compute class coverage statistics
4. **Change Detection**
   - Compare segmentation masks across years
   - Generate binary change map and compute gain/loss/net change
5. **Visualization + Summary**
   - Display original images, overlays, change maps, and narrative summaries  [oai_citation:9‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)

---

## Dataset
- **Region:** Chennai taluk (location-specific remote sensing dataset)
- **Time span:** 2014–2024
- **Total images:** 2,304 (≈192 per year)
- **Image size:** 512 × 512
- **Approx. resolution:** ≈ 30 m per pixel
- **Themes:** Vegetation, Water, Urban (and mentions of Barren in dataset description)  [oai_citation:10‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)

**Source:** Satellite images collected from **Google Earth Pro** (multi-year).  [oai_citation:11‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)

---

## Models / Methods Used
### 1) DistilBERT (Intent Detection)
- Fine-tuned classifier to map user queries to supported intent types.
- Chosen for lightweight inference + strong semantic understanding.  [oai_citation:12‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)

### 2) CLIP (Cross-modal Embeddings)
- Uses CLIP Text Encoder + Image Encoder to create aligned **512-dimensional embeddings**.
- Enables semantic matching beyond exact metadata keywords.  [oai_citation:13‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)

### 3) FAISS (Vector Retrieval)
- FAISS index (IndexFlatIP) built over normalized image embeddings.
- Fast Top-K similarity search using cosine similarity (via inner product).  [oai_citation:14‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)

### 4) DeepLabV3+ (Semantic Segmentation)
- Encoder–decoder architecture with ASPP for multi-scale context.
- Used to segment three land-cover classes: vegetation, water, urban.  [oai_citation:15‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)

### 5) Pixel-level Change Detection
- Binary differencing of theme masks across years.
- Computes %gain / %loss / net change and produces a change map.  [oai_citation:16‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)

---

## Results Snapshot (from report)
- Retrieval speed: **Top-K under ~0.5s on CPU** and ~0.1s on GPU (reported)
- Semantic retrieval accuracy reported higher than keyword baseline
- Segmentation improved across training (accuracy mentioned ~0.69; mIoU improvements reported)  [oai_citation:17‡final_external[1].pdf](sediment://file_0000000096047208b214654772b87c46)

---

## How to Run (Template)
> Update the commands below to match your repo (file names / entry script).  
> This section is intentionally written as a safe template because GitHub repos differ.

### 1) Create environment & install dependencies
```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
