"""
app.py (FINAL FIXED VERSION)
✅ Temporal + Visualization queries
✅ Enlarged Original + Mask images (side-by-side full width)
✅ No HTML/code block issues
✅ Professional Streamlit UI
"""

import streamlit as st
from chatbot import dispatch
from image_processing import ensure_real_model_loaded
import pandas as pd
from collections import defaultdict
import re
from rapidfuzz import fuzz

# ===============================================================
# Initialization
# ===============================================================
ensure_real_model_loaded()

st.set_page_config(
    page_title="🌍 Land-Cover Change Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================
# CSS Styling
# ===============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background: #f8fafc; padding: 2rem; }
.stApp { background: #f8fafc; }
h1 { color: #1e293b; font-size: 2.5rem !important; font-weight: 700 !important;
     text-align: center; padding: 1rem; margin-bottom: 2rem; letter-spacing: -0.02em; }
.stTextInput > div > div > input { background: white; border: 2px solid #e2e8f0;
    border-radius: 12px; padding: 16px 24px; font-size: 1rem; color: #1e293b;
    transition: all 0.3s ease; }
.stTextInput > div > div > input:focus { border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1); }
.stButton > button { background:#3b82f6; color:white; border:none; border-radius:12px;
    padding:14px 40px; font-size:1rem; font-weight:600; cursor:pointer;
    transition:all .3s ease; box-shadow:0 2px 8px rgba(59,130,246,.2);}
.stButton > button:hover{background:#2563eb;box-shadow:0 4px 12px rgba(59,130,246,.3);
    transform:translateY(-1px);}
.location-section{background:white;border-radius:16px;padding:32px;margin:32px 0;
    box-shadow:0 4px 16px rgba(0,0,0,.08);border:1px solid #e2e8f0;}
.location-header{color:#1e293b;font-size:1.75rem;font-weight:700;margin:0 0 24px 0;
    padding-bottom:16px;border-bottom:2px solid #e2e8f0;display:flex;align-items:center;
    gap:12px;}
.image-container{border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.1);
    transition:all .3s ease;background:white;border:1px solid #e2e8f0;margin-top:8px;}
.image-container:hover{box-shadow:0 6px 16px rgba(0,0,0,.15);transform:translateY(-4px);}
.year-badge{background:#f1f5f9;padding:8px 16px;border-radius:8px;font-weight:600;
    color:#475569;text-align:center;font-size:.9rem;border:1px solid #e2e8f0;
    display:inline-flex;align-items:center;gap:6px;justify-content:center;margin-bottom:8px;}
.summary-box{background:white;border-radius:16px;padding:28px;margin:24px 0;
    box-shadow:0 4px 16px rgba(0,0,0,.08);border:1px solid #e2e8f0;border-left:4px solid #3b82f6;}
.summary-box h3{color:#1e293b;margin-top:0;font-weight:700;}
.summary-box p{color:#475569;line-height:1.8;}
.divider{border:none;height:1px;background:#e2e8f0;margin:32px 0;}
span.top-badge{margin-left:10px;color:#2563eb;font-size:.9rem;font-weight:600;transition:.3s;}
span.top-badge:hover{color:#1d4ed8;transform:scale(1.05);}
</style>
""", unsafe_allow_html=True)

# ===============================================================
# Helper Functions
# ===============================================================
def extract_year(caption):
    if not caption:
        return None
    m = re.search(r'\b(20\d{2})\b', str(caption))
    return m.group(1) if m else None

def extract_location(caption):
    if not caption:
        return "Unknown"
    location = re.sub(r'\s*-?\s*20\d{2}\s*', '', str(caption))
    return location.replace('_', ' ').strip()

def extract_query_location_terms(query:str)->list:
    stop = {'show','display','visualize','vegetation','water','urban',
            'evolution','from','to','in','the','and','or'}
    words = re.findall(r'\b[a-z]+\b', query.lower())
    return [w for w in words if w not in stop and len(w)>2]

def compute_location_relevance(location_name:str, query_terms:list)->float:
    if not query_terms: return 0.0
    norm = location_name.lower().replace('_',' ')
    for t in query_terms:
        if t in norm: return 1.0
    qstr = ' '.join(query_terms)
    return fuzz.token_set_ratio(qstr, norm)/100.0

# ===============================================================
# Header and Sidebar
# ===============================================================
st.markdown("<h1>🌍 Temporal Land-Cover Change Analysis</h1>", unsafe_allow_html=True)
try:
    ensure_real_model_loaded()
    st.success("✅ DeepLabV3+ Segmentation Model Active")
except Exception as e:
    st.error(f"⚠️ Model Error: {e}")

with st.sidebar:
    st.markdown("### 🔍 Query Examples")
    st.markdown("""
    <div><h4>📅 Temporal Analysis</h4>
    <ul><li>Show vegetation evolution in Perambur from 2010 to 2025</li>
        <li>Animate urban growth in Guindy from 2015 to 2024</li></ul></div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div><h4>🗺️ Single Year</h4>
    <ul><li>Visualize vegetation coverage in Mathur 2022</li>
        <li>Show water in Guindy 2021</li>
        <li>Display vegetation in Kolathur 2020</li></ul></div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div><h4>💡 Tips</h4>
    <ul><li>Use full location names</li><li>Specify years clearly</li>
        <li>Mention theme: vegetation, water, or urban</li></ul></div>
    """, unsafe_allow_html=True)

# ===============================================================
# Main Interface
# ===============================================================
st.markdown("### 🔎 Enter Your Analysis Query")
user_query = st.text_input("", "Show vegetation evolution in Sembium from 2020 to 2025",
                           key="query_input", label_visibility="collapsed")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    analyze_button = st.button("🚀 Analyze", key="analyze_btn", use_container_width=True)

# ===============================================================
# Core Logic
# ===============================================================
if analyze_button:
    with st.spinner("🔄 Analyzing query and processing imagery..."):
        st.session_state.last_query = user_query
        result = dispatch(user_query)

    if isinstance(result, dict):
        st.markdown(f"""
        <div class='summary-box'>
            <h3>🧠 Analysis Summary</h3>
            <p>{result.get("text","")}</p>
        </div>""", unsafe_allow_html=True)

        results_list = result.get("results", [])
        if results_list:
            query_terms = extract_query_location_terms(user_query)
            all_images_by_location = defaultdict(list)

            # Group images by location
            for res in results_list:
                for img in res.get("temporal_images", []):
                    cap = img.get("caption","")
                    yr = extract_year(cap)
                    loc = extract_location(cap)
                    if loc and loc!="Unknown":
                        all_images_by_location[loc].append({
                            "image": img.get("image"),
                            "caption": cap,
                            "year": yr or "N/A",
                            "similarity_score": res.get("similarity_score",0),
                            "score_breakdown": img.get("score_breakdown")
                        })

                imgs, caps = res.get("images", []), res.get("captions", [])
                for i in range(0,len(imgs),2):
                    if i+1 < len(imgs):
                        yr = extract_year(caps[i]) if i < len(caps) else None
                        loc = extract_location(caps[i]) if i < len(caps) else "Unknown"
                        if loc and loc!="Unknown":
                            all_images_by_location[loc].append({
                                "images_pair":[imgs[i],imgs[i+1]],
                                "captions_pair":[caps[i],caps[i+1]],
                                "year": yr or "N/A",
                                "similarity_score": res.get("similarity_score",0),
                                "score_breakdown": res.get("score_breakdown")
                            })

            # Sort locations
            keys={}
            for loc,imgs in all_images_by_location.items():
                best=max([x.get("similarity_score",0) for x in imgs],default=0)
                rel=compute_location_relevance(loc,query_terms)
                keys[loc]=(-best,-rel,loc.lower())
            sorted_locs=sorted(keys.keys(),key=lambda x:keys[x])

            # Display each location
            for loc in sorted_locs:
                imgs=all_images_by_location[loc]
                imgs.sort(key=lambda x:int(x["year"]) if str(x["year"]).isdigit() else 9999)

                best=max([i.get("similarity_score",0) for i in imgs],default=0)
                rel=compute_location_relevance(loc,query_terms)
                score=None
                for i in imgs:
                    if i.get("score_breakdown") is not None:
                        score=i["score_breakdown"]; break

                top=(loc==sorted_locs[0])
                indicator="🎯" if rel>0.8 else "📍"

                # header
                if score:
                    score_html = (
                        f"<div style='font-size:.85rem;color:#64748b;margin-top:4px;'>"
                        f"<strong>Match:</strong> {best:.3f} | <strong>Relevance:</strong> {rel:.2f}</div>"
                    )
                else:
                    score_html = f"<div style='font-size:.85rem;color:#64748b;'>Match: {best:.3f} | Relevance: {rel:.2f}</div>"

                st.markdown(f"""
                <div class='location-section'>
                    <div class='location-header'>
                        {indicator} {loc} {'<span class="top-badge">⭐ Top Match</span>' if top else ''}
                    </div>
                    <div style='margin-left:2rem;'>{score_html}</div>
                </div>
                """, unsafe_allow_html=True)

                # ======================================================
                # IMAGE RENDERING SECTION (Fixed)
                # ======================================================
                has_pairs = any(("images_pair" in item) for item in imgs)
                if has_pairs:
                    # Visualization query — big side-by-side display
                    for img in imgs:
                        if "images_pair" in img and isinstance(img["images_pair"], list):
                            year = img.get("year", "N/A")
                            st.markdown(f"<div class='year-badge'>📅 {year}</div>", unsafe_allow_html=True)

                            original, mask = img["images_pair"]
                            captions = img.get("captions_pair", ["Original", "Mask"])

                            st.markdown("<div style='margin-top:12px; margin-bottom:20px;'>", unsafe_allow_html=True)
                            left, right = st.columns([1, 1], gap="large")
                            with left:
                                st.image(original, caption=f"🛰️ {captions[0]}", use_container_width=True)
                            with right:
                                st.image(mask, caption=f"🌿 {captions[1]}", use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                else:
                    # Temporal query — grid layout
                    cols_per_row = 4
                    for i in range(0, len(imgs), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for c, img in zip(cols, imgs[i:i + cols_per_row]):
                            with c:
                                year = img.get("year", "N/A")
                                st.markdown(f"<div class='year-badge'>📅 {year}</div>", unsafe_allow_html=True)
                                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                                st.image(img["image"], caption=img.get("caption", ""), use_container_width=True)
                                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        else:
            st.info("ℹ️ No visualization data available for this query.")
    elif isinstance(result,str):
        st.info(f"ℹ️ {result}")
    else:
        st.warning("⚠️ Unable to process query. Please try rephrasing.")

# ===============================================================
# Footer
# ===============================================================
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;padding:32px;color:#64748b;'>
  <p style='font-size:.95rem;margin:8px 0;color:#475569;'>
    <strong style='color:#1e293b;'>Powered by:</strong> DeepLabV3+ Segmentation | CLIP Embeddings | FAISS Retrieval
  </p>
  <p style='font-size:.85rem;color:#94a3b8;'>
    🌍 Temporal Land-Cover Analysis System | Built with Streamlit
  </p>
</div>
""", unsafe_allow_html=True)
