"""
Quick Setup Guide for Auto-AVSR (Better Lip Reading Model)

This model works on NEW videos (not just training data)!
"""

# Installation steps:

# 1. Open PowerShell and run:
cd c:\Users\VAIBHAV RAGHAV\Desktop\Lip2
git clone https://github.com/mpc001/auto_avsr.git
cd auto_avsr

# 2. Activate your virtual environment
& "c:\Users\VAIBHAV RAGHAV\Desktop\Lip2\myenv\Scripts\Activate.ps1"

# 3. Install requirements
pip install -r requirements.txt

# 4. Download pre-trained model (this will download ~500MB)
# Follow instructions in the repository's README

# 5. Test on your videos!

"""
Key Differences from Current LipNet:

✅ Works on NEW, UNSEEN videos (not overfitted)
✅ Better accuracy (~40% error vs 60%+ error)
✅ Trained on 1000+ hours of diverse speakers
✅ Modern transformer architecture
✅ Active development and support

❌ Slightly slower (but more accurate)
❌ Larger model size (~500MB vs 50MB)
"""
