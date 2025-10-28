# How to Add Your Own Videos to LipNet

## Quick Start

### Step 1: Prepare Your Video
Your video should have:
- ✅ Clear view of the person's face (frontal view)
- ✅ Good lighting on the face
- ✅ Visible mouth/lip movements
- ✅ Supported formats: `.mp4`, `.mpg`, `.avi`, or `.mov`

### Step 2: Add Video to the App
Simply copy your video file to:
```
LipNet/data/s1/
```

Example:
```
LipNet/data/s1/myvideo.mp4
LipNet/data/s1/test_speech.mpg
```

### Step 3: Run the App
The app will automatically detect your new videos!

```powershell
cd "c:\Users\VAIBHAV RAGHAV\Desktop\Lip2\LipNet\app"
& "C:/Users/VAIBHAV RAGHAV/Desktop/Lip2/myenv/Scripts/streamlit.exe" run streamlitapp.py
```

## That's It! 🎉

Your videos will appear in the dropdown menu and the AI will make predictions on them.

## Optional: Add Ground Truth Text

If you want to compare the AI predictions with the actual spoken text:

1. Create a file: `LipNet/data/alignments/s1/myvideo.align`
2. Format (time in milliseconds, then word):
```
0 1000 sil
1000 2000 hello
2000 3000 world
3000 4000 sil
```

Note: The alignment file is **optional**. The app will work without it!

## Tips for Best Results

1. **Face Position**: Person should be facing the camera directly
2. **Lighting**: Even, bright lighting on the face
3. **Resolution**: Higher resolution = better results
4. **No Obstructions**: No hands, masks, or other objects covering the mouth
5. **Clear Articulation**: Clear lip movements work best

## Troubleshooting

- **Video not showing?** Make sure it's in `LipNet/data/s1/` folder
- **No predictions?** Check that the face is visible and well-lit
- **Format error?** Convert your video to `.mp4` or `.mpg` format
