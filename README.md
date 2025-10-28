# Silent Speak

Silent Speak — a lip-reading / silent-speech recognition project.

## Overview
This repo implements a LipNet-style 3D-CNN + Bidirectional LSTM model using TensorFlow/Keras and CTC loss. It includes preprocessing scripts, training code, a CTC decoder, and Docker config so you can reproduce experiments.

## Highlights
- 3D Conv layers to extract spatiotemporal features
- Bidirectional LSTM for sequence modeling
- CTC loss & decoder for alignment-free sequence labeling
- Scripts for preprocessing GRID-like datasets (crop mouth, grayscale, normalize)

