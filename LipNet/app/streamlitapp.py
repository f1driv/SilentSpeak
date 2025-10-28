# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np

import tensorflow as tf
import tf_keras as keras
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App') 
# Generating a list of options or videos 
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 's1')
all_files = os.listdir(data_path)
# Filter .mpg and .mp4 video files and sort them
options = sorted([f for f in all_files if f.endswith(('.mpg', '.mp4', '.avi', '.mov'))])
st.info(f'📹 {len(options)} videos available for testing')
selected_video = st.selectbox('Choose video', options)

# Audio control in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("🔊 Audio Controls")
    play_audio = st.checkbox("Enable Audio Playback", value=True)
    st.info("Toggle to hear what the person is saying in the video")

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Show selected video info
    st.success(f"🎬 Currently viewing: **{selected_video}**")
    
    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video with audio')
        file_path = os.path.join(data_path, selected_video)
        
        # Use unique filename based on selected video to avoid caching
        # Remove file extension (works for .mpg, .mp4, .avi, etc.)
        video_name = os.path.splitext(selected_video)[0]
        output_video = f'test_video_{video_name}.mp4'
        
        # Convert video with or without audio based on user preference
        if play_audio:
            # Convert with audio
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 -acodec aac {output_video} -y 2>nul')
        else:
            # Convert without audio (muted)
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 -an {output_video} -y 2>nul')

        # Rendering inside of the app
        if os.path.exists(output_video):
            with open(output_video, 'rb') as video_file:
                video_bytes = video_file.read() 
            st.video(video_bytes, start_time=0)
        else:
            # If ffmpeg is not available, show the original video
            st.warning("FFmpeg not found. Showing original .mpg file (audio may not work in all browsers)")
            with open(file_path, 'rb') as video_file:
                video_bytes = video_file.read()
            # Use key parameter to force Streamlit to reload the video when selection changes
            st.video(video_bytes, start_time=0)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        # Convert normalized float32 frames to uint8 for imageio
        # The video is z-score normalized, so we need to rescale it to 0-255
        video_min = tf.reduce_min(video)
        video_max = tf.reduce_max(video)
        video_normalized = (video - video_min) / (video_max - video_min)
        video_for_display = (video_normalized * 255).numpy().astype('uint8')
        # Squeeze the last dimension (channel dimension) since it's grayscale
        video_for_display = video_for_display.squeeze()
        # Use unique filename for GIF based on selected video
        gif_path = f'animation_{video_name}.gif'
        imageio.mimsave(gif_path, video_for_display, fps=10)
        st.image(gif_path, width=400)
        
        # Audio playback option
        if play_audio:
            st.markdown("### 🎧 Audio Only")
            # Extract audio from video with unique filename
            audio_path = f'audio_{video_name}.mp3'
            result = os.system(f'ffmpeg -i {file_path} -vn -acodec libmp3lame -q:a 2 {audio_path} -y 2>nul')
            if result == 0 and os.path.exists(audio_path):
                with open(audio_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')
            else:
                st.caption("Audio extraction requires FFmpeg to be installed")

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
