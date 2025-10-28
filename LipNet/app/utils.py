import tensorflow as tf
from tf_keras.layers import StringLookup
from typing import List
import cv2
import os 

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path:str) -> List[float]: 
    #print(path)
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        if not ret:
            break
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    # Convert to tensor
    frames = tf.convert_to_tensor(frames)
    
    # Handle videos of different lengths - model expects exactly 75 frames
    num_frames = tf.shape(frames)[0]
    target_frames = 75
    
    if num_frames > target_frames:
        # If video is too long, take evenly spaced frames
        indices = tf.cast(tf.linspace(0, num_frames - 1, target_frames), tf.int32)
        frames = tf.gather(frames, indices)
    elif num_frames < target_frames:
        # If video is too short, pad with last frame
        padding_needed = target_frames - num_frames
        last_frame = frames[-1:]
        padding = tf.tile(last_frame, [padding_needed, 1, 1, 1])
        frames = tf.concat([frames, padding], axis=0)
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std
    
def load_alignments(path:str) -> List[str]: 
    #print(path)
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    # Get file name and extension
    file_name_with_ext = path.split('/')[-1]
    # File name splitting for windows
    file_name_with_ext = path.split('\\')[-1]
    file_name = file_name_with_ext.split('.')[0]
    file_ext = file_name_with_ext.split('.')[-1]
    
    # Use absolute path - the path passed in should already be absolute from streamlit
    video_path = path
    
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    frames = load_video(video_path)
    
    # Try to load alignment file, but make it optional
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alignment_path = os.path.join(script_dir, '..', 'data', 'alignments', 's1', f'{file_name}.align')
    if os.path.exists(alignment_path):
        alignments = load_alignments(alignment_path)
    else:
        # Return empty alignments if file doesn't exist (for custom videos)
        alignments = char_to_num(tf.reshape(tf.strings.unicode_split('', input_encoding='UTF-8'), (-1)))
    
    return frames, alignments