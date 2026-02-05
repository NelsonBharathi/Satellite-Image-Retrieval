# # """
# # debug_test.py
# # ------------------------------------
# # Diagnostic script to validate full chatbot + model pipeline
# # ✔ Checks model load mode (real vs mock)
# # ✔ Tests retrieval + visualization per theme
# # ✔ Verifies mask statistics and color overlays
# # ✔ Saves overlay results for visual inspection
# # """

# # import pandas as pd
# # import numpy as np
# # from pathlib import Path
# # import torch
# # import cv2
# # import os

# # OUTPUT_DIR = Path("debug_outputs")
# # OUTPUT_DIR.mkdir(exist_ok=True)

# # print("=" * 70)
# # print("STEP 1: Verify Image Manifest")
# # print("=" * 70)

# # manifest_path = Path("data/processed/image_manifest.csv")
# # if manifest_path.exists():
# #     df = pd.read_csv(manifest_path)
# #     print(f"✅ Loaded image manifest: {len(df)} rows, {df['location'].nunique()} unique locations")
# #     print(df.head(3))
# # else:
# #     print("❌ Manifest not found. Please ensure path is correct.")
# #     exit()

# # print("\n" + "=" * 70)
# # print("STEP 2: Import and Check Model Status")
# # print("=" * 70)

# # try:
# #     import image_processing as ip

# #     if not ip.SEGMENTATION_MODELS:
# #         print("⚠️ Mock mode active — no real model loaded.")
# #     else:
# #         model_name = list(ip.SEGMENTATION_MODELS.keys())[0]
# #         device = next(ip.SEGMENTATION_MODELS[model_name].parameters()).device
# #         print(f"✅ Real model '{model_name}' loaded on {device}")
# #         print(f"Model class: {type(ip.SEGMENTATION_MODELS[model_name]).__name__}")
# # except Exception as e:
# #     print(f"❌ Error importing image_processing: {e}")
# #     import traceback
# #     traceback.print_exc()
# #     exit()

# # print("\n" + "=" * 70)
# # print("STEP 3: Single Image Segmentation Check")
# # print("=" * 70)

# # test_image = Path("F:/images/2020/Guindy_Adyar_(Part_I).png")
# # if not test_image.exists():
# #     print(f"❌ Test image not found: {test_image}")
# #     exit()
# # else:
# #     print(f"✅ Found test image: {test_image}")

# # # Test across all 3 themes
# # themes = ["vegetation", "water", "urban"]
# # for theme in themes:
# #     print(f"\n🧩 Testing segmentation for theme: {theme}")
# #     try:
# #         orig, overlay, mask = ip.get_mask_overlay(str(test_image), theme=theme)
# #         print(f" - Original shape: {orig.shape}")
# #         print(f" - Overlay shape: {overlay.shape}")
# #         print(f" - Mask shape: {mask.shape}")
# #         print(f" - Mask unique values: {sorted(set(mask.flatten().tolist()))}")

# #         # Class-wise statistics
# #         for class_id, class_name in {0:"background",1:"vegetation",2:"water",3:"urban"}.items():
# #             pixels = np.sum(mask == class_id)
# #             pct = pixels / mask.size * 100
# #             print(f"   • {class_name:11s}: {pct:.2f}%")

# #         # Save overlay for verification
# #         out_path = OUTPUT_DIR / f"overlay_{theme}.png"
# #         cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
# #         print(f"   💾 Saved overlay: {out_path}")

# #     except Exception as e:
# #         print(f"❌ Error while processing theme '{theme}': {e}")

# # print("\n" + "=" * 70)
# # print("STEP 4: Chatbot Full Query Test")
# # print("=" * 70)

# # try:
# #     from chatbot import dispatch
# #     query = "Show vegetation evolution in Adyar from 2020 to 2024"
# #     print(f"Query: {query}\n")
# #     result = dispatch(query)

# #     if isinstance(result, dict):
# #         print(f"✅ Dispatch returned dict with keys: {list(result.keys())}")
# #         print(f"📄 Summary text:\n{result.get('text', '')}")
# #         if result.get("results"):
# #             for res in result["results"]:
# #                 print(f"\n📍 Location: {res.get('location')}")
# #                 imgs = res.get("temporal_images", [])
# #                 print(f"   Temporal images: {len(imgs)}")
# #                 if imgs:
# #                     print(f"   Example: {imgs[0]['caption']}")
# #         else:
# #             print("⚠️ No temporal results in output.")
# #     else:
# #         print(f"Unexpected result type: {type(result)}\n{result}")

# # except Exception as e:
# #     print(f"❌ Error running chatbot dispatch: {e}")
# #     import traceback
# #     traceback.print_exc()

# # print("\n" + "=" * 70)
# # print("STEP 5: Model Confirmation")
# # print("=" * 70)

# # if ip.SEGMENTATION_MODELS:
# #     model = list(ip.SEGMENTATION_MODELS.values())[0]
# #     param_count = sum(p.numel() for p in model.parameters())
# #     print(f"✅ Model verified: {type(model).__name__}")
# #     print(f"   Total parameters: {param_count:,}")
# #     print(f"   Device: {next(model.parameters()).device}")
# # else:
# #     print("⚠️ Model verification skipped (mock mode active).")

# # print("\n" + "=" * 70)
# # print("✅ DIAGNOSTIC COMPLETE")
# # print("=" * 70)
# # print(f"All overlay images saved in: {OUTPUT_DIR.resolve()}")



# """
# location_debug.py
# ----------------------------------
# Batch diagnostic tool for verifying:
# - Location detection (from chatbot)
# - Fuzzy matching behavior
# - FAISS retrieval filtering
# - End-to-end results

# Run this file directly:
#     python location_debug.py
# """

# import pandas as pd
# import re
# from pathlib import Path

# print("=" * 80)
# print("🌍 LAND-COVER CHATBOT — LOCATION DEBUG TOOL")
# print("=" * 80)

# # Step 1: Import chatbot + retrieval
# try:
#     import chatbot
#     from multimodal_retrieval import MultimodalRetrieval
#     print("✅ Successfully imported chatbot and retrieval modules.")
# except Exception as e:
#     print(f"❌ Import error: {e}")
#     raise SystemExit

# # Step 2: Initialize retrieval engine (with debug enabled)
# try:
#     retrieval = MultimodalRetrieval()
#     retrieval.DEBUG_MATCHING = True  # enable verbose logs
# except Exception as e:
#     print(f"⚠️ Retrieval engine init failed: {e}")
#     retrieval = None

# # Step 3: Prepare test queries
# queries = [
#     "Show vegetation in Alandhur 2020",
#     "Show vegetation in Alandur 2020",
#     "Display water in Sholinganallur 2019",
#     "Animate urban growth in Guindy from 2015 to 2020",
#     "Compare vegetation in Purasawalkam between 2018 and 2023",
#     "Show vegetation evolution in Mountroad from 2020 to 2024",
#     "Display vegetation in Kolathur 2021",
#     "Visualize water in Aminjikarai 2020",
# ]

# print("\n🧠 Testing queries:")
# for i, q in enumerate(queries, 1):
#     print(f"  {i:02d}. {q}")
# print("=" * 80)

# # Step 4: Run queries and analyze detection
# for query in queries:
#     print("\n" + "-" * 80)
#     print(f"📝 Query: {query}")

#     try:
#         # Extract internal details
#         intent = chatbot.detect_intent(query)
#         theme = chatbot.detect_theme(query)
#         y0, y1 = chatbot.detect_years(query)
#         locations = chatbot.detect_location(query)

#         print(f"🔍 Intent: {intent}")
#         print(f"🔍 Theme: {theme}")
#         print(f"🔍 Years: {y0}, {y1}")
#         print(f"📍 Detected Locations: {locations if locations else '❌ None detected'}")

#         # Retrieval test if location found
#         if locations and retrieval is not None:
#             for loc in locations:
#                 print(f"\n➡ Checking retrieval match for: '{loc}'")
#                 df = retrieval.metadata_df
#                 subset = df[
#                     (df["location"].str.lower().str.contains(loc.lower()))
#                     | (df["location_normalized"].str.lower().str.contains(loc.lower()))
#                 ]
#                 print(f"   - Found {len(subset)} matching entries in metadata.")
#                 if not subset.empty:
#                     for _, row in subset.head(3).iterrows():
#                         print(f"     🔹 {row['location']} ({row['year']})")

#             print("\n🧩 Testing semantic search (top-3):")
#             result = retrieval.search(query_text=f"{theme} cover", top_k=3, location_filter=locations[0])
#             if not result:
#                 print("   ⚠️ No results found for retrieval filter.")
#             else:
#                 for r in result:
#                     print(f"   ✅ Match: {r['location']} ({r['year']}) | Score={r['similarity_score']:.3f}")

#         else:
#             print("⚠️ Skipping retrieval — no location detected.")

#     except Exception as e:
#         print(f"❌ Error for query '{query}': {e}")

# print("\n" + "=" * 80)
# print("✅ DEBUG SESSION COMPLETE")
# print("=" * 80)



"""
location_debug_v2.py
--------------------------------------------------
Advanced diagnostic for Alandhur mismatch + fuzzy thresholds.
Tests multiple queries and computes fuzzy similarity for all metadata entries.
"""

import re
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz, process

print("=" * 90)
print("🧭 MULTI-QUERY LOCATION DEBUG — FUZZY MATCH THRESHOLD EXPLORER")
print("=" * 90)

# Load modules
try:
    import chatbot
    from multimodal_retrieval import MultimodalRetrieval
    print("✅ Imported chatbot and retrieval successfully.")
except Exception as e:
    print(f"❌ Import failed: {e}")
    raise SystemExit

# Initialize retrieval
retrieval = MultimodalRetrieval()
meta = retrieval.metadata_df
meta["location_combined"] = (
    meta["location"].astype(str)
    + " "
    + meta["location_normalized"].astype(str)
    + " "
    + meta["taluk_norm"].astype(str)
    + " "
    + meta["locality_norm"].astype(str)
)

# Queries to test
queries = [
    "Show vegetation evolution in alandhur from 2020 to 2024",
    "Show vegetation change in alandur between 2018 and 2024",
    "Analyze water bodies in guindy 2020 to 2025",
    "Visualize vegetation cover in purasawalkam voc nagar 2021",
    "Show vegetation trends in kolathur 2020",
    "Check water evolution in aminjikarai 2020",
]

# Fuzzy threshold levels to test
THRESHOLDS = [50, 60, 70, 80, 85, 90]

def top_matches(word, df, top_n=5):
    """Return top fuzzy matches for a location term."""
    results = []
    for text in df["location_combined"]:
        score = fuzz.token_set_ratio(word.lower(), text.lower())
        results.append((text, score))
    return sorted(results, key=lambda x: x[1], reverse=True)[:top_n]


for query in queries:
    print("\n" + "-" * 90)
    print(f"🧠 Query: {query}")

    intent = chatbot.detect_intent(query)
    theme = chatbot.detect_theme(query)
    y0, y1 = chatbot.detect_years(query)
    locs = chatbot.detect_location(query)

    print(f"🔍 Intent: {intent} | Theme: {theme} | Years: {y0}-{y1}")
    print(f"📍 Extracted locations: {locs if locs else '❌ None detected'}")

    if not locs:
        continue

    for loc in locs:
        print(f"\n📌 Checking fuzzy matches for '{loc}':")
        matches = top_matches(loc, meta, top_n=10)
        for name, score in matches:
            print(f"   - {name[:60]:<60} | score={score:.1f}")

        # Evaluate thresholds
        best_match = matches[0][0] if matches else None
        best_score = matches[0][1] if matches else 0
        print(f"   🧩 Best match: '{best_match}' ({best_score:.1f})")

        for t in THRESHOLDS:
            flag = "✅" if best_score >= t else "❌"
            print(f"      Threshold {t}: {flag}")

    # Optional: try a retrieval call if fuzzy score ≥ 80
    best_loc = locs[0] if locs else ""
    results = retrieval.search(theme, top_k=3, location_filter=best_loc)
    if results:
        print("\n📸 Retrieval results (top 3):")
        for r in results:
            print(f"   - {r['image_path']} | {r['location']} ({r['year']}) | Score={r['similarity_score']:.3f}")
    else:
        print("⚠️ No retrieval results found (likely fuzzy mismatch).")

print("\n" + "=" * 90)
print("✅ MULTI-QUERY LOCATION DEBUG COMPLETE")
print("=" * 90)
