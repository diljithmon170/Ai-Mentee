import os
import time
from transformers import pipeline
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
import pysrt

# Function to generate video from text and audio
def generate_video_from_subject_level(subject_name, level):
    text_file_path = f"{subject_name}-{level}.txt"
    audio_file_path = f"{subject_name}-{level}.mp3"
    video_file_path = f"{subject_name}-{level}.mp4"

    # Check if files exist
    if not os.path.exists(text_file_path):
        print(f"Error: Text file '{text_file_path}' not found.")
        return
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file '{audio_file_path}' not found.")
        return

    # Read text content
    with open(text_file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Placeholder for text-to-video model
    print(f"Generating video for {subject_name}-{level}...")
    video = None  # Replace with an actual text-to-video model call

    # Save the video file (dummy placeholder)
    if video:
        with open(video_file_path, "wb") as video_file:
            video_file.write(video)
    
    # Process the audio
    audio = AudioSegment.from_file(audio_file_path)
    audio_wav_path = f"{subject_name}-{level}.wav"
    audio.export(audio_wav_path, format="wav")

    # Combine video and audio
    video_clip = VideoFileClip(video_file_path)
    audio_clip = AudioFileClip(audio_wav_path)
    final_video = video_clip.set_audio(audio_clip)
    final_video.write_videofile(video_file_path, codec="libx264")

    # Generate and add subtitles
    subtitles_path = generate_subtitles(audio_wav_path, subject_name, level)
    add_subtitles_to_video(video_file_path, subtitles_path)

# Function to generate subtitles using ASR
def generate_subtitles(audio_file_path, subject_name, level):
    print("Generating subtitles...")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large")

    # Transcribe audio
    transcription = asr(audio_file_path)["text"]

    # Save subtitles
    subtitles_path = f"{subject_name}-{level}-subtitles.srt"
    with open(subtitles_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    return subtitles_path

# Function to add subtitles to video
def add_subtitles_to_video(video_file_path, subtitles_path):
    video_clip = VideoFileClip(video_file_path)
    subs = pysrt.open(subtitles_path)

    subtitle_clips = []
    for sub in subs:
        subtitle_text_clip = TextClip(sub.text, fontsize=24, color='white', bg_color='black', size=(video_clip.w, 50))
        subtitle_text_clip = subtitle_text_clip.set_start(sub.start.seconds).set_end(sub.end.seconds)
        subtitle_text_clip = subtitle_text_clip.set_position(('center', 'bottom'))
        subtitle_clips.append(subtitle_text_clip)

    final_video = CompositeVideoClip([video_clip] + subtitle_clips)
    final_video_path = video_file_path.replace(".mp4", "-final.mp4")
    final_video.write_videofile(final_video_path, codec="libx264")

    print(f"Final video saved at {final_video_path}")


