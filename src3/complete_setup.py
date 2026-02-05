"""
complete_setup.py
Automated Setup Script
Runs all necessary steps to build a fully functional system
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed."""
    required = [
        'streamlit',
        'torch',
        'transformers',
        'sentence-transformers',
        'faiss-cpu',
        'segmentation-models-pytorch',
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'Pillow'
    ]
    
    print("=" * 70)
    print("CHECKING DEPENDENCIES")
    print("=" * 70)
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All dependencies installed")
    return True

def setup_directories():
    """Create necessary directory structure."""
    print("\n" + "=" * 70)
    print("SETTING UP DIRECTORY STRUCTURE")
    print("=" * 70)
    
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "models/segmentation_models",
        "models/intent_classifier",
        "models/faiss_index"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {d}")

def run_data_preparation():
    """Run data preparation pipeline."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPARATION")
    print("=" * 70)
    
    try:
        from data_preparation import run_data_prep
        run_data_prep()
        print("\n✅ Data preparation complete")
        return True
    except Exception as e:
        print(f"\n❌ Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_training():
    """Train intent classifier."""
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING INTENT CLASSIFIER")
    print("=" * 70)
    
    try:
        from training import run_training
        run_training()
        print("\n✅ Training complete")
        return True
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def build_faiss_index():
    """Build FAISS index for semantic retrieval."""
    print("\n" + "=" * 70)
    print("STEP 3: BUILDING FAISS INDEX")
    print("=" * 70)
    print("⏳ This may take 10-30 minutes for large image collections...")
    
    try:
        from multimodal_retrieval import MultimodalRetrieval
        
        retrieval = MultimodalRetrieval()
        
        # Check if index already exists
        if retrieval.index_path.exists():
            user_input = input("\nFAISS index already exists. Rebuild? (y/n): ")
            if user_input.lower() != 'y':
                print("Skipping FAISS index building")
                return True
        
        retrieval.build_index()
        
        print("\n✅ FAISS index built successfully")
        return True
    except Exception as e:
        print(f"\n❌ FAISS indexing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_system():
    """Validate that all components are working."""
    print("\n" + "=" * 70)
    print("STEP 4: SYSTEM VALIDATION")
    print("=" * 70)
    
    checks = {
        "Image Manifest": "data/processed/image_manifest.csv",
        "Synthetic Queries": "data/processed/synth_queries.csv",
        "Intent Model": "models/intent_classifier",
        "Label Encoder": "models/label_encoder.joblib",
        "FAISS Index": "models/faiss_index",
        "FAISS Metadata": "data/processed/embeddings_metadata.csv"
    }
    
    all_valid = True
    for name, path in checks.items():
        if Path(path).exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} - NOT FOUND")
            all_valid = False
    
    return all_valid

def test_chatbot():
    """Run basic chatbot tests."""
    print("\n" + "=" * 70)
    print("STEP 5: TESTING CHATBOT")
    print("=" * 70)
    
    try:
        # Import after setup is complete
        import chatbot
        
        test_queries = [
            "Show vegetation in guindy for 2020",
            "List urban areas in aminjikarai",
            "Compare water in adyar 2015 2020"
        ]
        
        print("\nRunning test queries...")
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            try:
                response = chatbot.dispatch(query)
                if isinstance(response, dict):
                    num_results = len(response.get('results', []))
                    print(f"   ✅ Response type: dict with {num_results} results")
                else:
                    print(f"   ✅ Response: {response[:100]}...")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n✅ Chatbot tests complete")
        return True
    except Exception as e:
        print(f"\n❌ Chatbot testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_quick_start_guide():
    """Create a quick start guide."""
    guide = """
# Temporal Land Cover QA Chatbot - Quick Start Guide

## ✅ System Setup Complete!

All components have been installed and configured.

## 🚀 How to Run

### Start the Chatbot
```bash
streamlit run app.py
```

The browser should open automatically at http://localhost:8501

## 📝 Example Queries

**Visualization (Single Year):**
- "Show vegetation in Guindy for 2020"
- "Display water coverage in Aminjikarai 2018"

**Temporal Analysis (Multi-Year):**
- "Animate urban growth in Guindy from 2015 to 2020"
- "Show vegetation evolution in Adyar from 2010 to 2025"

**Descriptive Analysis:**
- "List vegetation in Guindy"
- "Describe water bodies in Aminjikarai"

**Comparative Analysis:**
- "Compare water in Aminjikarai 2015 vs 2020"
- "Urban change in Guindy between 2018 and 2023"

## 🔧 System Architecture

Your system now includes:

1. **Query Processing** → Intent + NER + Embeddings
2. **Retrieval Engine** → FAISS semantic search
3. **Image Analysis** → Segmentation + Change Detection
4. **Response Generation** → Text summaries + Visualizations

## 🐛 Troubleshooting

**No images found:**
- Check that image_manifest.csv has correct paths
- Verify images exist at specified locations
- Update image directory path in data_preparation.py

**FAISS errors:**
- Rebuild index: `python -c "from multimodal_retrieval import MultimodalRetrieval; MultimodalRetrieval().build_index()"`

**Intent classification issues:**
- Retrain model: `python training.py`

## 📞 Quick Reference Commands

```bash
# Start application
streamlit run app.py

# Rebuild FAISS index
python -c "from multimodal_retrieval import MultimodalRetrieval; MultimodalRetrieval().build_index()"

# Retrain intent classifier
python training.py

# Recreate image manifest
python data_preparation.py

# Run full setup again
python complete_setup.py
```

## 🎉 You're Ready!

Your Temporal Land Cover QA Chatbot is now fully operational with:

- ✅ Semantic image search (CLIP + FAISS)
- ✅ Intent classification (DistilBERT)
- ✅ Land cover segmentation
- ✅ Temporal change detection
- ✅ Natural language interface

Enjoy exploring your satellite imagery!
"""
    
    with open("QUICKSTART.md", "w") as f:
        f.write(guide)
    
    print("\n📖 Quick Start Guide created: QUICKSTART.md")

def main():
    """Main setup orchestration."""
    print("\n" + "🚀" * 35)
    print("TEMPORAL LAND COVER QA CHATBOT - COMPLETE SETUP")
    print("🚀" * 35)
    
    # Step 0: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        print("Run: pip install -r requirements.txt")
        return
    
    # Step 1: Setup directories
    setup_directories()
    
    # Step 2: Data preparation
    print("\n⏳ Starting data preparation...")
    if not run_data_preparation():
        print("\n⚠️ Warning: Data preparation had issues")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            return
    
    # Step 3: Training
    print("\n⏳ Starting intent classifier training...")
    if not run_training():
        print("\n⚠️ Warning: Training had issues")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            return
    
    # Step 4: Build FAISS index
    print("\n⏳ Starting FAISS index building...")
    if not build_faiss_index():
        print("\n⚠️ Warning: FAISS indexing had issues")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            return
    
    # Step 5: Validation
    valid = validate_system()
    
    # Step 6: Testing
    if valid:
        print("\n⏳ Testing chatbot...")
        test_chatbot()
    
    # Step 7: Create documentation
    create_quick_start_guide()
    
    # Final summary
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    
    if valid:
        print("\n✅ System is ready to use!")
        print("\n🚀 To start the chatbot:")
        print("   streamlit run app.py")
        print("\n📖 See QUICKSTART.md for usage guide")
    else:
        print("\n⚠️ Setup completed with warnings")
        print("Check error messages above and fix issues")
        print("You can re-run: python complete_setup.py")

if __name__ == "__main__":
    main()