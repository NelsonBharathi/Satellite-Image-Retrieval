"""
data_preparation.py
Data Preparation Module
Creates image manifest and synthetic training queries
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define constants
THEMES = {
    "water": ["water", "river", "lake"], 
    "vegetation": ["tree", "forest", "green"], 
    "urban": ["urban", "building", "city"]
}

def create_image_manifest(image_dir="F:/images"):
    """
    Scans the images directory and creates a manifest file.
    
    Expected structure:
        images/
        ├── 2015/
        │   ├── Location1.png
        │   └── Location2.png
        ├── 2016/
        │   └── Location1.png
    
    Creates manifest with normalized location names for easier matching.
    """
    print(f"Scanning {image_dir} directory...")
    
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        print(f"⚠️ Warning: Directory '{image_dir}' not found")
        df = pd.DataFrame(columns=['location', 'location_normalized', 'year', 'image_path'])
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        df.to_csv("data/processed/image_manifest.csv", index=False)
        return df
    
    # Find all PNG and JPG images recursively
    image_paths = list(image_dir_path.glob("**/*.png")) + list(image_dir_path.glob("**/*.jpg"))
    
    if not image_paths:
        print(f"⚠️ Warning: No images found in {image_dir}/ directory")
        df = pd.DataFrame(columns=['location', 'location_normalized', 'year', 'image_path'])
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        df.to_csv("data/processed/image_manifest.csv", index=False)
        return df
    
    # Extract metadata from each image path
    manifest_data = []
    skipped = []
    
    for p in image_paths:
        try:
            # Original filename (e.g., "Aminjikarai_Chinnakudal")
            location_original = p.stem
            
            # Normalized version for matching (lowercase, replace underscores with spaces)
            location_normalized = location_original.lower().replace('_', ' ')
            
            # Extract year from parent folder name
            try:
                year = int(p.parent.name)
            except ValueError:
                # Skip if folder name is not a valid year
                skipped.append(str(p))
                continue
            
            # Full path
            image_path = str(p)
            
            manifest_data.append({
                "location": location_original,  # Keep original for display
                "location_normalized": location_normalized,  # For matching queries
                "year": year,
                "image_path": image_path
            })
        except Exception as e:
            skipped.append(f"{p}: {e}")
            continue
    
    if skipped:
        print(f"⚠️ Skipped {len(skipped)} files (parent folder not a year):")
        for s in skipped[:5]:  # Show first 5
            print(f"   - {s}")
        if len(skipped) > 5:
            print(f"   ... and {len(skipped)-5} more")
    
    # Create DataFrame and save
    df = pd.DataFrame(manifest_data)
    
    if not df.empty:
        df = df.sort_values(['location_normalized', 'year'])  # Sort for easier reading
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        df.to_csv("data/processed/image_manifest.csv", index=False)
        
        # Print summary
        print(f"\n✅ Image manifest created with {len(df)} entries")
        print(f"  - Locations: {df['location_normalized'].nunique()} unique")
        print(f"  - Year range: {df['year'].min()} to {df['year'].max()}")
        print(f"  - Sample locations:")
        for loc in df['location'].unique()[:5]:
            print(f"    • {loc}")
        if df['location'].nunique() > 5:
            print(f"    ... and {df['location'].nunique()-5} more")
        print(f"  - Saved to: data/processed/image_manifest.csv")
    else:
        print("⚠️ No valid images found to create manifest")
    
    return df

def synth_queries(taluks, localities, n=5):
    """Generates synthetic queries for all intents."""
    np.random.seed(42)
    rows = []
    
    # Templates for different intents
    list_t = [
        "list {theme} in {place}",
        "what {theme} exists in {place}",
        "describe {theme} coverage in {place}"
    ]
    diff_t = [
        "{theme} change in {place} 2020 2025",
        "compare {theme} in {place} between 2015 and 2020",
        "difference in {theme} for {place} 2018 2023"
    ]
    viz_t = [
        "show mask for {theme} in {place} 2021", 
        "display {theme} in {place} for 2020",
        "visualize {theme} coverage in {place} 2022"
    ]
    temp_viz_t = [
        "animate {theme} in {place} from 2015 to 2020", 
        "show {theme} cover of {place} from 2010 to 2015",
        "temporal view of {theme} in {place} 2018 to 2023"
    ]
    
    places = taluks + localities

    for place in places:
        for _ in range(n):
            theme = np.random.choice(list(THEMES.keys()))
            rand_val = np.random.rand()
            
            if rand_val < 0.25:
                q = np.random.choice(list_t).format(theme=theme, place=place)
                intent = "descriptive"
            elif rand_val < 0.5:
                q = np.random.choice(diff_t).format(theme=theme, place=place)
                intent = "analytical"
            elif rand_val < 0.75:
                q = np.random.choice(viz_t).format(theme=theme, place=place)
                intent = "visualize"
            else:
                q = np.random.choice(temp_viz_t).format(theme=theme, place=place)
                intent = "temporal_visualize"
            
            rows.append((q, intent, theme, place))

    df = pd.DataFrame(rows, columns=["query", "intent", "theme", "place"])
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/synth_queries.csv", index=False)
    print(f"✅ Synthetic queries generated: {len(df)} queries")
    print(f"\nIntent distribution:")
    print(df['intent'].value_counts())
    return df

def run_data_prep():
    """Main function to run all data preparation steps."""
    print("=" * 70)
    print("STARTING DATA PREPARATION")
    print("=" * 70)
    
    # Create directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # Load raw data
    try:
        df_multi = pd.read_csv("data/raw/multi_embeddings_output.csv")
        print(f"✅ Loaded multi_embeddings_output.csv: {len(df_multi)} rows")
    except FileNotFoundError:
        print("⚠️ multi_embeddings_output.csv not found, using defaults")
        df_multi = pd.DataFrame({"landmark": ["guindy", "aminjikarai", "adyar"]})
    
    try:
        df_syn = pd.read_csv("data/raw/synthetic_landcover_taluk.csv")
        print(f"✅ Loaded synthetic_landcover_taluk.csv: {len(df_syn)} rows")
    except FileNotFoundError:
        print("⚠️ synthetic_landcover_taluk.csv not found, using defaults")
        df_syn = pd.DataFrame({"Taluk": ["guindy", "aminjikarai", "t_nagar"]})
    
    # Extract unique locations
    taluks = sorted(df_syn["Taluk"].dropna().astype(str).unique())
    localities = sorted(df_multi["landmark"].dropna().astype(str).unique())
    
    print(f"✅ Found {len(taluks)} taluks and {len(localities)} localities")
    
    # Generate synthetic queries
    synth_queries(taluks, localities, n=5)
    
    # Create image manifest
    create_image_manifest()
    
    print("=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    run_data_prep()