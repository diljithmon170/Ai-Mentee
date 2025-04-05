import os
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
from transformers import pipeline
from pydub import AudioSegment
import ffmpeg

def generate_video_with_audio_and_subtitles(subject, level, number):
    text_file = f"{subject}-{level}-{number}.txt"
    audio_file = f"{subject}-{level}-{number}.wav"
    video_file = f"{subject}-{level}-{number}.mp4"
    final_video_file = f"{subject}-{level}-{number}.mp4"  # Auto-save as requested

    # Check if input files exist
    if not os.path.exists(text_file):
        print(f"Error: Text file '{text_file}' not found.")
        return
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        return

    # Read LLaMA-generated text
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Generate video using CogVideoX
   
    video_generator = pipeline("text-to-video", model="cog-video/cogvideo-5b")
    video_data = video_generator(text)
    with open(video_file, "wb") as f:
        f.write(video_data)

    # Load audio
    audio = AudioSegment.from_wav(audio_file)
    audio_duration = len(audio) / 1000  # Convert to seconds

    # Load video
    video_clip = VideoFileClip(video_file)
    
    # Ensure video matches audio duration
    if video_clip.duration < audio_duration:
        
        video_clip = video_clip.loop(duration=audio_duration)
    else:
        video_clip = video_clip.subclip(0, audio_duration)  # Trim if video is longer
    
    # Set audio to video
    video_clip = video_clip.set_audio(AudioFileClip(audio_file))

    # Generate subtitles
    
    words = text.split()
    subtitle_clips = []
    words_per_second = len(words) / audio_duration if words else 1

    # Create timed subtitles
    current_time = 0
    for word in words:
        subtitle = TextClip(word, fontsize=36, color='white', bg_color='black', font="Arial-Bold")
        subtitle = subtitle.set_start(current_time).set_duration(1 / words_per_second).set_position(('center', 'bottom'))
        subtitle_clips.append(subtitle)
        current_time += 1 / words_per_second

    # Overlay subtitles on video
    final_video = CompositeVideoClip([video_clip] + subtitle_clips)

    # Export final video
    
    final_video.write_videofile(final_video_file, codec="libx264", fps=24, threads=4)

    










