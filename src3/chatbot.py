"""
chatbot.py - ENHANCED VERSION
Passes query context and similarity scores for intelligent ranking
"""

import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Segmentation
import image_processing as ip
from image_processing import get_mask_overlay, ensure_real_model_loaded

print("🔧 Chatbot Segmentation Models Loaded at import:", list(ip.SEGMENTATION_MODELS.keys()))
ensure_real_model_loaded()

# Retrieval
try:
    from multimodal_retrieval import MultimodalRetrieval, ChangeDetector
    RETRIEVAL_ENABLED = True
except ImportError:
    print("⚠️ Multimodal retrieval not available; using manifest only.")
    MultimodalRetrieval = None
    ChangeDetector = None
    RETRIEVAL_ENABLED = False

# Globals
IMAGE_MANIFEST: pd.DataFrame | None = None
INTENT_TOKENIZER = None
INTENT_MODEL = None
LABEL_ENCODER = None
LOCATIONS: list[str] = []
THEMES = ["vegetation", "water", "urban"]
RETRIEVAL_ENGINE: MultimodalRetrieval | None = None
CHANGE_DETECTOR: ChangeDetector | None = None

THEME_TO_CLASS = {"vegetation": 1, "water": 2, "urban": 3}


def initialize():
    """Load all models and initialize retrieval engine."""
    global IMAGE_MANIFEST, INTENT_TOKENIZER, INTENT_MODEL, LABEL_ENCODER
    global LOCATIONS, RETRIEVAL_ENGINE, CHANGE_DETECTOR

    print("=" * 70)
    print("INITIALIZING ENHANCED LAND COVER CHATBOT")
    print("=" * 70)

    ensure_real_model_loaded()
    if ip.SEGMENTATION_MODELS and "deeplab" in ip.SEGMENTATION_MODELS:
        print("✅ Real DeepLabV3+ model active in chatbot:", list(ip.SEGMENTATION_MODELS.keys()))
    else:
        print("⚠️ DeepLabV3+ model NOT found; check image_processing model path.")

    # Load Image Manifest
    manifest_paths = [
        "data/processed/image_manifest.csv",
        "data/image_manifest.csv",
        "image_manifest.csv",
    ]
    df = None
    for p in manifest_paths:
        pth = Path(p)
        if pth.exists():
            try:
                df = pd.read_csv(pth)
                print(f"✅ Loaded image manifest from {p}")
                break
            except Exception as e:
                print(f"⚠️ Failed reading {p}: {e}")

    if df is None or df.empty:
        print("⚠️ Manifest not found or empty.")
        IMAGE_MANIFEST = pd.DataFrame(columns=["location", "location_normalized", "year", "image_path"])
        LOCATIONS = []
    else:
        df["image_path"] = df["image_path"].astype(str)
        df["location"] = df["location"].astype(str)

        def parse_year(row):
            y = row.get("year", None)
            if pd.notna(y):
                try:
                    return int(y)
                except Exception:
                    pass
            m = re.search(r"[\\/](20\d{2})[\\/]", row["image_path"])
            return int(m.group(1)) if m else None

        df["year"] = df.apply(parse_year, axis=1)
        df = df[df["year"].notna()].copy()
        if df.empty:
            print("⚠️ Manifest had no valid years after parsing.")
        else:
            df["year"] = df["year"].astype(int)

        df["location_normalized"] = (
            df["location"]
            .str.replace("_", " ", regex=False)
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        def split_norm(s: str):
            toks = s.split()
            taluk = toks[0] if len(toks) >= 1 else ""
            locality = toks[1] if len(toks) >= 2 else ""
            return taluk, locality

        pairs = df["location_normalized"].apply(split_norm)
        df["taluk_norm"] = pairs.apply(lambda x: x[0])
        df["locality_norm"] = pairs.apply(lambda x: x[1])

        IMAGE_MANIFEST = df.reset_index(drop=True)
        LOCATIONS = sorted(df["location"].unique())
        print(f"✅ Manifest processed: {len(df)} rows, {len(LOCATIONS)} unique locations")

    # Intent Model
    try:
        INTENT_TOKENIZER = DistilBertTokenizer.from_pretrained("./models/intent_classifier")
        INTENT_MODEL = DistilBertForSequenceClassification.from_pretrained("./models/intent_classifier")
        LABEL_ENCODER = joblib.load("./models/label_encoder.joblib")
        INTENT_MODEL.eval()
        print("✅ Loaded DistilBERT intent classifier.")
    except Exception as e:
        print(f"⚠️ Intent model unavailable, using rule-based fallback: {e}")

    # Retrieval Engine
    if RETRIEVAL_ENABLED and MultimodalRetrieval is not None:
        try:
            print("🔧 Initializing Multimodal Retrieval Engine...")
            RETRIEVAL_ENGINE = MultimodalRetrieval()
            print("✅ Retrieval engine initialized.")
        except Exception as e:
            print(f"⚠️ Retrieval init failed: {e}")
            RETRIEVAL_ENGINE = None
    else:
        RETRIEVAL_ENGINE = None

    # Change Detector
    try:
        CHANGE_DETECTOR = ChangeDetector()
        print("✅ Change detector ready.")
    except Exception as e:
        print(f"⚠️ Change detector unavailable: {e}")
        CHANGE_DETECTOR = None

    print("=" * 70)
    print("INITIALIZATION COMPLETE")
    print("=" * 70)


def _ensure_text(x):
    if isinstance(x, str):
        return x
    return str(x) if x is not None else ""


def detect_intent(q: str) -> str:
    """Enhanced intent detection."""
    q = _ensure_text(q).lower()
    
    has_range = bool(re.search(r"from\s+20\d{2}\s+to\s+20\d{2}", q))
    has_evolution = any(w in q for w in ["animate", "temporal", "evolution", "change over time", "trend"])
    
    if has_evolution or has_range:
        return "temporal_visualize"
    
    if any(w in q for w in ["show", "display", "visualize", "map", "mask"]):
        return "visualize"
    
    if any(w in q for w in ["change", "compare", "difference", "contrast"]):
        return "analytical"
    
    if any(w in q for w in ["describe", "list", "what", "find", "tell me about"]):
        return "descriptive"
    
    return "unknown"


def detect_theme(q: str) -> str:
    """Theme detection."""
    q = _ensure_text(q).lower()
    if "water" in q or "river" in q or "lake" in q:
        return "water"
    if "urban" in q or "building" in q or "built" in q or "city" in q:
        return "urban"
    return "vegetation"


def detect_years(q: str) -> tuple[int | None, int | None]:
    """Extract years from query."""
    q = _ensure_text(q)
    yrs = re.findall(r"20(1[0-9]|2[0-5])", q)
    if not yrs:
        return (None, None)
    ys = sorted([int(f"20{y}") for y in yrs])
    return (ys[0], ys[-1]) if len(ys) > 1 else (ys[0], ys[0])


try:
    from rapidfuzz import process, fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    print("⚠️ rapidfuzz not installed; location matching will be simpler.")
    _HAS_RAPIDFUZZ = False


def detect_location(q: str):
    """Enhanced location detection with better precision."""
    if IMAGE_MANIFEST is None or IMAGE_MANIFEST.empty:
        return None

    q_clean = re.sub(r"[^a-z0-9\s]", " ", _ensure_text(q).lower()).strip()
    q_words = set(q_clean.split())

    taluks = IMAGE_MANIFEST["taluk_norm"].dropna().unique().tolist()
    localities = IMAGE_MANIFEST["locality_norm"].dropna().unique().tolist()
    locations = IMAGE_MANIFEST["location"].unique().tolist()

    matches = set()

    # Priority 1: Exact substring match for localities
    for l in localities:
        if l and len(l) > 2 and l in q_clean:
            locs = IMAGE_MANIFEST[IMAGE_MANIFEST["locality_norm"] == l]["location"].unique()
            matches.update(locs)

    # Priority 2: Taluk match
    if not matches:
        for t in taluks:
            if t and len(t) > 2 and t in q_clean:
                locs = IMAGE_MANIFEST[IMAGE_MANIFEST["taluk_norm"] == t]["location"].unique()
                matches.update(locs)

    # Priority 3: Fuzzy matching
    if not matches and _HAS_RAPIDFUZZ:
        pool = [l for l in localities if l] + [t for t in taluks if t]
        try:
            best_matches = process.extract(q_clean, pool, scorer=fuzz.token_set_ratio, limit=3)
            for best, score, _ in best_matches:
                if score >= 75:  # Lowered threshold
                    locs = IMAGE_MANIFEST[
                        (IMAGE_MANIFEST["taluk_norm"] == best) |
                        (IMAGE_MANIFEST["locality_norm"] == best)
                    ]["location"].unique()
                    matches.update(locs)
        except Exception as e:
            print(f"⚠️ Fuzzy matching error: {e}")

    # Priority 4: Word-level matching
    if not matches:
        for loc in locations:
            loc_words = set(loc.lower().replace("_", " ").split())
            if q_words & loc_words:
                matches.add(loc)

    result = sorted(matches) if matches else None
    if result:
        print(f"✅ Detected locations: {result}")
    return result


def manifest_based_retrieval(locations: list[str] | None, year: int | None):
    """Retrieve images from manifest with default score of 1.0."""
    results = []
    if IMAGE_MANIFEST is None or IMAGE_MANIFEST.empty or not locations or year is None:
        return results
    
    for loc in locations:
        df = IMAGE_MANIFEST[
            (IMAGE_MANIFEST["location"] == loc) & (IMAGE_MANIFEST["year"] == year)
        ]
        for _, row in df.iterrows():
            results.append({
                "image_path": row["image_path"],
                "location": row["location"],
                "year": int(row["year"]),
                "similarity_score": 1.0  # Default score for manifest-based retrieval
            })
    
    print(f"📊 Manifest retrieval: Found {len(results)} images for {locations} in {year}")
    return results


def semantic_search(theme: str, locations: list[str] | None, year: int | None, top_k: int = 5):
    """Semantic search with fallback to manifest."""
    if RETRIEVAL_ENGINE is None:
        print("⚠️ Retrieval engine not available, using manifest")
        return manifest_based_retrieval(locations, year)

    results = []
    for loc in (locations or []):
        loc_clean = re.sub(r"[^a-z0-9\s]", " ", loc.lower())
        try:
            r = RETRIEVAL_ENGINE.search(
                query_text=f"{theme} coverage in {loc}",
                top_k=top_k,
                location_filter=loc_clean,
                year_filter=year
            )
            if r:
                results.extend(r)
                print(f"🔍 Semantic search for {loc}: Found {len(r)} images with scores: {[x.get('similarity_score', 0) for x in r[:3]]}")
        except Exception as e:
            print(f"⚠️ Retrieval search failed for '{loc}': {e}")
            results.extend(manifest_based_retrieval([loc], year))
    
    if not results:
        print("⚠️ Semantic search returned nothing, falling back to manifest")
        results = manifest_based_retrieval(locations, year)
    
    return results


def visualize_with_retrieval(theme: str, locations: list[str], year: int, original_query: str = ""):
    """
    Enhanced: Return results with similarity_score and original query context
    """
    print(f"🔍 Visualizing {theme} for {locations} ({year})")
    
    results = semantic_search(theme, locations, year, top_k=10)
    
    if not results:
        return {
            "text": f"No results found for {', '.join(locations)} in {year}", 
            "results": None,
            "original_query": original_query
        }

    out = []
    for r in results[:5]:  # Process top 5 results
        try:
            original, overlay, _mask = get_mask_overlay(r["image_path"], theme=theme)
            
            out.append({
                "location": r.get("location", ""),
                "similarity_score": r.get("similarity_score", 0.0),
                "score_breakdown": r.get("score_breakdown", None),  # Pass through breakdown
                "images": [original, overlay],
                "captions": [
                    f"{r.get('location', '')} - Original ({year})",
                    f"{r.get('location', '')} - {theme.capitalize()} Mask ({year})"
                ]
            })
            print(f"✅ Processed {r.get('location')} with score {r.get('similarity_score', 0):.3f}")
        except Exception as e:
            print(f"⚠️ Error processing {r.get('image_path')}: {e}")

    return {
        "text": f"Found {len(out)} images for {theme} in {', '.join(locations)} ({year})",
        "results": out,
        "original_query": original_query
    }


def temporal_visualize_with_change(theme: str, locations: list[str], start_year: int, end_year: int, original_query: str = ""):
    """
    Enhanced: Return temporal_images with similarity_score and query context
    """
    print(f"🔍 Temporal: {theme}, {start_year}-{end_year}, {locations}")
    results = []

    # Try semantic search first
    if RETRIEVAL_ENGINE:
        for loc in (locations or []):
            try:
                r = RETRIEVAL_ENGINE.temporal_search(
                    query_text=f"{theme} coverage",
                    location=loc,
                    start_year=start_year,
                    end_year=end_year
                )
                if r:
                    results.extend(r)
                    print(f"🔍 Temporal search for {loc}: Found {len(r)} images")
            except Exception as e:
                print(f"⚠️ Retrieval temporal_search failed for '{loc}': {e}")

    # Fallback to manifest
    if not results and IMAGE_MANIFEST is not None and not IMAGE_MANIFEST.empty:
        for loc in (locations or []):
            df = IMAGE_MANIFEST[
                (IMAGE_MANIFEST["location"] == loc) &
                (IMAGE_MANIFEST["year"].between(start_year, end_year, inclusive="both"))
            ].sort_values("year")
            
            # Add default similarity score
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict["similarity_score"] = 1.0  # Default score
                results.append(row_dict)
            
            print(f"📊 Manifest temporal for {loc}: Found {len(df)} images")

    if not results:
        return {
            "text": f"No images found for {', '.join(locations)} between {start_year}-{end_year}.", 
            "results": None,
            "original_query": original_query
        }

    # Sort by year
    def _safe_year(x):
        try:
            return int(x.get("year", 0))
        except Exception:
            return 0

    results = sorted(results, key=_safe_year)

    temporal_images, masks = [], []
    for r in results:
        try:
            _, overlay, mask = get_mask_overlay(r["image_path"], theme=theme)
            masks.append(mask)
            
            temporal_images.append({
                "image": overlay,
                "caption": f"{r.get('location','')} - {int(r.get('year',0))}",
                "similarity_score": r.get("similarity_score", 1.0)
            })
            print(f"✅ Processed {r.get('location')} ({r.get('year')}) with score {r.get('similarity_score', 1.0):.3f}")
        except Exception as e:
            print(f"⚠️ Error processing {r.get('image_path')}: {e}")

    change_text = ""
    if len(masks) >= 2 and CHANGE_DETECTOR is not None:
        class_id = THEME_TO_CLASS.get(theme, 1)
        try:
            _change_map, stats = CHANGE_DETECTOR.compute_binary_change(masks[0], masks[-1], class_id=class_id)
            change_text = (
                f"\n\n**Change Detection:**\n"
                f"• Net: {stats['net_change_percent']:+.2f}%\n"
                f"• Loss: {stats['loss_percent']:.2f}%\n"
                f"• Gain: {stats['gain_percent']:.2f}%"
            )
        except Exception as e:
            print(f"⚠️ Change detection failed: {e}")

    # Calculate average similarity score for this location
    avg_score = sum(img.get("similarity_score", 1.0) for img in temporal_images) / len(temporal_images) if temporal_images else 0.0

    return {
        "text": f"Temporal change for {theme} ({start_year}-{end_year})" + change_text,
        "results": [{
            "location": ", ".join(locations),
            "temporal_images": temporal_images,
            "similarity_score": avg_score
        }],
        "original_query": original_query
    }


def dispatch(q):
    """Main dispatcher with improved logic and query context preservation."""
    if not isinstance(q, str):
        q = _ensure_text(q)
    q = q.strip()
    if not q:
        return "⚠️ Please enter a valid query."

    ensure_real_model_loaded()

    intent = detect_intent(q)
    theme = detect_theme(q)
    y0, y1 = detect_years(q)
    locations = detect_location(q)

    print(f"🔍 Intent: {intent} | Theme: {theme} | Years: {y0}-{y1} | Locations: {locations}")

    if not locations:
        some = ", ".join(LOCATIONS[:5]) if LOCATIONS else "no-locations-in-manifest"
        return f"⚠️ Location not recognized. Try one of: {some}"

    # Pass original query for context-aware ranking
    if intent == "visualize" and y0:
        return visualize_with_retrieval(theme, locations, y0, original_query=q)
    
    elif intent == "temporal_visualize" and y0 and y1:
        return temporal_visualize_with_change(theme, locations, y0, y1, original_query=q)
    
    elif intent == "descriptive":
        if IMAGE_MANIFEST is None or IMAGE_MANIFEST.empty:
            return f"No data available for {', '.join(locations)}."
        
        records = IMAGE_MANIFEST[IMAGE_MANIFEST["location"].isin(locations)]
        if records.empty:
            return f"No data for {', '.join(locations)}."
        
        years = sorted(set(records["year"].tolist()))
        return f"Data available for {', '.join(locations)} from {years[0]} to {years[-1]}."
    
    elif intent == "analytical":
        if not (y0 and y1):
            return "Please specify two years to compare (e.g., 2018 and 2024)."
        return f"Analytical comparison queued for {theme} in {', '.join(locations)}: {y0} vs {y1}."
    
    else:
        return f"Intent '{intent}' detected. Please specify a year or range (e.g., 'Show {theme} in Guindy 2020' or 'from 2020 to 2024')."


# Initialize on import
initialize()