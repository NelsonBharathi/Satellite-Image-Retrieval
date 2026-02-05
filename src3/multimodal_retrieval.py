"""
multimodal_retrieval.py - VERIFIED VERSION
Ensures FAISS similarity scores are correctly returned
"""

import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Tuple
from rapidfuzz import fuzz

FUZZY_LOC_THRESHOLD = 75  # Lowered for better recall


def safe_lower(x):
    if isinstance(x, str):
        return x.lower()
    try:
        return str(x).lower()
    except Exception:
        return ""


def norm_text(s: str) -> str:
    """Normalize text by removing symbols and excess whitespace."""
    s = safe_lower(s).replace("_", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class MultimodalRetrieval:
    """FAISS + CLIP retrieval with proper similarity score handling."""
    
    def __init__(self, index_path="models/faiss_index", metadata_path="data/processed/embeddings_metadata.csv"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Initializing Multimodal Retrieval on {self.device}")

        print("📦 Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        print("📦 Loading SBERT model...")
        self.text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.faiss_index = None
        self.metadata_df = None
        self.embedding_dim = 512
        self.DEBUG_MATCHING = False

        if self.index_path.exists() and self.metadata_path.exists():
            self.load_index()

    def encode_image_clip(self, image_path: str) -> np.ndarray:
        """Encode image using CLIP."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vec = self.clip_model.get_image_features(**inputs)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy().flatten()

    def encode_text_clip(self, text: str) -> np.ndarray:
        """Encode text using CLIP."""
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            vec = self.clip_model.get_text_features(**inputs)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy().flatten()

    def build_index(self, image_manifest_path="data/processed/image_manifest.csv"):
        """Build FAISS index with Inner Product (for cosine similarity)."""
        print("🗃️ Building FAISS index from image manifest...")
        m = pd.read_csv(image_manifest_path)

        # CRITICAL: Use IndexFlatIP for cosine similarity (vectors are normalized)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        embs, records = [], []

        for _, row in m.iterrows():
            p = row.get("image_path", "")
            if not Path(p).exists():
                print(f"⚠️ Missing image: {p}")
                continue
            try:
                emb = self.encode_image_clip(p)
                embs.append(emb)
                records.append({
                    "image_path": p,
                    "location": str(row.get("location", "")),
                    "location_normalized": str(row.get("location_normalized", "")),
                    "year": row.get("year", None),
                })
                if len(embs) % 50 == 0:
                    print(f"  Processed {len(embs)} images...")
            except Exception as e:
                print(f"⚠️ {p}: {e}")

        if not embs:
            print("❌ No embeddings to index.")
            return

        # Add to FAISS index
        embeddings_array = np.vstack(embs).astype("float32")
        self.faiss_index.add(embeddings_array)

        # Create metadata
        df = pd.DataFrame(records)
        df["location_normalized"] = df["location_normalized"].map(norm_text)

        def split_norm(s):
            toks = s.split()
            taluk = toks[0] if len(toks) >= 1 else ""
            locality = toks[1] if len(toks) >= 2 else ""
            return taluk, locality

        tloc = df["location_normalized"].apply(split_norm)
        df["taluk_norm"] = tloc.apply(lambda x: x[0])
        df["locality_norm"] = tloc.apply(lambda x: x[1])

        df = df.fillna("").astype(str)
        df.to_csv(self.metadata_path, index=False)
        self.metadata_df = df

        self.save_index()
        print(f"✅ Index size: {self.faiss_index.ntotal} | Meta: {len(self.metadata_df)} rows")

    def save_index(self):
        """Save FAISS index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(self.index_path))
        print(f"💾 Saved FAISS → {self.index_path}")

    def load_index(self):
        """Load FAISS index from disk."""
        self.faiss_index = faiss.read_index(str(self.index_path))
        self.metadata_df = pd.read_csv(self.metadata_path).fillna("").astype(str)
        print(f"✅ Loaded FAISS ({self.faiss_index.ntotal}) and metadata ({len(self.metadata_df)})")

    def _match_location(self, meta: dict, loc_filter: str) -> bool:
        """Enhanced location matching with multiple strategies."""
        if not loc_filter:
            return True

        lf = norm_text(loc_filter)
        
        fields = " ".join([
            norm_text(meta.get("location", "")),
            norm_text(meta.get("location_normalized", "")),
            norm_text(meta.get("taluk_norm", "")),
            norm_text(meta.get("locality_norm", "")),
        ])
        
        # Strategy 1: Substring match
        if lf in fields:
            if self.DEBUG_MATCHING:
                print(f"[MATCH✅] Substring: '{lf}' in '{fields}'")
            return True
        
        # Strategy 2: Word overlap
        lf_words = set(lf.split())
        field_words = set(fields.split())
        if lf_words & field_words:
            if self.DEBUG_MATCHING:
                print(f"[MATCH✅] Word overlap: {lf_words & field_words}")
            return True
        
        # Strategy 3: Fuzzy matching
        score = fuzz.token_set_ratio(lf, fields)
        
        if self.DEBUG_MATCHING:
            print(f"[DEBUG] Fuzzy: '{lf}' ↔ '{fields}' => score={score}")
        
        return score >= FUZZY_LOC_THRESHOLD

    def search(self, query_text: str, top_k: int = 5, location_filter: str = None, year_filter: int = None) -> List[Dict]:
        """
        CRITICAL: Properly return similarity scores from FAISS Inner Product search.
        For normalized vectors with IndexFlatIP, scores are cosine similarity in range [-1, 1].
        """
        if self.faiss_index is None or self.metadata_df is None:
            print("❌ Index not ready.")
            return []

        # Encode query
        q = self.encode_text_clip(query_text).reshape(1, -1).astype("float32")
        
        # FAISS search returns (scores, indices)
        # With IndexFlatIP and normalized vectors, scores ARE cosine similarity
        scores, idxs = self.faiss_index.search(q, self.faiss_index.ntotal)

        print(f"🔍 FAISS search returned {len(scores[0])} results")
        print(f"   Score range: [{scores[0].min():.4f}, {scores[0].max():.4f}]")

        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            
            meta = self.metadata_df.iloc[idx].to_dict()
            
            # Apply filters
            if location_filter and not self._match_location(meta, location_filter):
                continue
            
            if year_filter is not None:
                try:
                    y = int(float(meta.get("year", "0")))
                except Exception:
                    y = None
                if y != year_filter:
                    continue
            
            # CRITICAL: Store the actual similarity score
            # Normalize to [0, 1] range: (score + 1) / 2
            normalized_score = float((score + 1) / 2)
            meta["similarity_score"] = normalized_score
            out.append(meta)
            
            if len(out) >= top_k:
                break
        
        if out:
            print(f"✅ Returning {len(out)} results with scores: {[r['similarity_score'] for r in out[:3]]}")
        else:
            print("⚠️ No results matched filters")
        
        return out

    def temporal_search(self, query_text: str, location: str, start_year: int, end_year: int) -> List[Dict]:
        """Temporal search with proper score handling."""
        if self.faiss_index is None or self.metadata_df is None:
            return []

        q = self.encode_text_clip(query_text).reshape(1, -1).astype("float32")
        scores, idxs = self.faiss_index.search(q, self.faiss_index.ntotal)

        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            
            meta = self.metadata_df.iloc[idx].to_dict()
            
            if not self._match_location(meta, location):
                continue
            
            try:
                y = int(float(meta.get("year", "0")))
            except Exception:
                y = None
            
            if y is None or y < start_year or y > end_year:
                continue
            
            # Normalize score to [0, 1]
            normalized_score = float((score + 1) / 2)
            meta["similarity_score"] = normalized_score
            out.append(meta)

        # Sort by year
        out.sort(key=lambda x: int(float(x.get("year", "0"))) if str(x.get("year", "")).strip() else 0)
        
        print(f"🔍 Temporal search: {len(out)} results for '{location}' ({start_year}-{end_year})")
        return out


class ChangeDetector:
    """Binary change detection between segmentation masks."""
    
    @staticmethod
    def compute_binary_change(mask1: np.ndarray, mask2: np.ndarray, class_id: int = None) -> Tuple[np.ndarray, Dict]:
        """Compute binary change map and statistics."""
        if mask1.shape != mask2.shape:
            mask2 = np.array(Image.fromarray(mask2).resize((mask1.shape[1], mask1.shape[0]), Image.NEAREST))

        if class_id is not None:
            binary1 = (mask1 == class_id).astype(np.uint8)
            binary2 = (mask2 == class_id).astype(np.uint8)
        else:
            binary1, binary2 = mask1, mask2

        change_map = np.zeros_like(binary1)
        change_map[(binary1 == 1) & (binary2 == 0)] = 1  # Loss
        change_map[(binary1 == 0) & (binary2 == 1)] = 2  # Gain

        total = mask1.size
        stats = {
            "total_pixels": int(total),
            "unchanged_percent": float(np.sum(change_map == 0) / total * 100),
            "loss_percent": float(np.sum(change_map == 1) / total * 100),
            "gain_percent": float(np.sum(change_map == 2) / total * 100),
        }
        stats["net_change_percent"] = stats["gain_percent"] - stats["loss_percent"]
        return change_map, stats


def integrate_multimodal_retrieval():
    """Initialize and return retrieval system."""
    retrieval = MultimodalRetrieval()
    if not retrieval.index_path.exists():
        print("Building FAISS index for the first time...")
        retrieval.build_index()
    return retrieval


if __name__ == "__main__":
    print("=" * 70)
    print("MULTIMODAL RETRIEVAL SYSTEM - SETUP")
    print("=" * 70)

    retrieval = MultimodalRetrieval()
    retrieval.DEBUG_MATCHING = True
    retrieval.build_index()

    # Test search
    results = retrieval.search("vegetation cover", top_k=3)
    print(f"\n🔍 Search results: {len(results)} images found")
    for r in results:
        print(f"  - {r['image_path']} (score: {r['similarity_score']:.3f})")

    print("\n✅ Multimodal retrieval system ready!")