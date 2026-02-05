#!/usr/bin/env python3
"""
Setup script for Running Shoe Video Classifier.

Run this to verify your installation and configure paths.

Usage: python setup.py
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor} (need 3.9+)")
        return False

def check_dependencies():
    """Check required packages."""
    print("\nChecking dependencies...")
    
    packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'scipy': 'scipy'
    }
    
    optional = {
        'mediapipe': 'mediapipe (for gesture detection)'
    }
    
    missing = []
    
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name}")
            missing.append(package_name)
    
    print("\nOptional dependencies:")
    for import_name, description in optional.items():
        try:
            __import__(import_name)
            print(f"  ✅ {description}")
        except ImportError:
            print(f"  ⚠️  {description} (not installed)")
    
    return missing

def check_paths():
    """Check configured paths."""
    print("\nChecking paths...")
    
    # Import config
    try:
        from config import VIDEO_DIR, find_models_dir, OUTPUT_DIR
        
        # Check video directory
        if VIDEO_DIR.exists():
            videos = list(VIDEO_DIR.glob("*.mp4"))
            print(f"  ✅ VIDEO_DIR: {VIDEO_DIR}")
            print(f"     Found {len(videos)} .mp4 files")
        else:
            print(f"  ❌ VIDEO_DIR not found: {VIDEO_DIR}")
            print("     Update VIDEO_DIR in config.py")
        
        # Check models directory
        models_dir = find_models_dir()
        if models_dir.exists():
            models = list(models_dir.glob("*.pth"))
            print(f"  ✅ MODELS_DIR: {models_dir}")
            print(f"     Found {len(models)} .pth files")
            for m in models:
                print(f"       - {m.name}")
        else:
            print(f"  ⚠️  MODELS_DIR not found: {models_dir}")
            print("     Models are optional - app will still work")
        
        # Check output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ OUTPUT_DIR: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"  ❌ Error importing config: {e}")

def check_pytorch_device():
    """Check PyTorch device availability."""
    print("\nChecking PyTorch devices...")
    
    try:
        import torch
        
        print(f"  PyTorch version: {torch.__version__}")
        
        # CPU always available
        print("  ✅ CPU available")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠️  CUDA not available")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  ✅ MPS (Apple Silicon) available")
        else:
            print("  ⚠️  MPS not available")
        
        # Show which device will be used
        from config import DEVICE
        print(f"\n  Selected device: {DEVICE}")
        
    except Exception as e:
        print(f"  ❌ Error checking devices: {e}")

def main():
    """Run all checks."""
    print("=" * 60)
    print("Running Shoe Video Classifier - Setup Check")
    print("=" * 60)
    
    # Python version
    check_python_version()
    
    # Dependencies
    missing = check_dependencies()
    
    if missing:
        print("\n⚠️  Missing packages. Install with:")
        print(f"   pip install {' '.join(missing)}")
        print("   Or: pip install -r requirements.txt")
    
    # PyTorch device
    check_pytorch_device()
    
    # Paths
    check_paths()
    
    print("\n" + "=" * 60)
    print("Setup check complete!")
    print("=" * 60)
    
    if not missing:
        print("\n✅ Ready to run! Start with:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  Install missing packages first, then run:")
        print("   streamlit run app.py")

if __name__ == "__main__":
    main()
