import os
import re
import io
import gc
import cv2
import time
import base64
import random
import tempfile
import subprocess

import numpy as np
import runpod

from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoClip, AudioFileClip


# ============================================================
# VIDEO SETTINGS
# ============================================================

WIDTH = 1920
HEIGHT = 1080
FPS = 30

VIDEO_BITRATE = "4000k"

BACKGROUND = (255, 255, 255)

# Character position
CHARACTER_X = 40
CHARACTER_Y = 100
CHARACTER_WIDTH = 760
CHARACTER_HEIGHT = 900

# Subtitle panel position
PANEL_X = 850
PANEL_Y = 250
PANEL_WIDTH = 990

# Subtitle settings
MAX_VISIBLE_CHUNKS = 4

CURRENT_COLOR = (255, 204, 0)
PREVIOUS_COLOR = (150, 150, 150)
TEXT_COLOR = (35, 35, 35)

# Animation settings
ANIMATION_SAMPLE_RATE = 0.10
MIN_POSE_DURATION = 0.65
GESTURE_INTERVAL = 18.0


# ============================================================
# CHARACTER ASSETS
# ============================================================

ASSET_DIR = "/app"

CHARACTER_FILES = {
    "neutral": "01_neutral.png",
    "speaking": "02_mouth_slightly_open.png",
    "wide": "03_mouth_wide_open.png",
    "gesture": "09_both_hands_gesture.png",
}


# ============================================================
# FONT SETTINGS
# ============================================================

def find_font(candidates, default=None):

    for path in candidates:

        if os.path.exists(path):
            return path

    return default


HINDI_FONT_PATH = find_font([
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagariUI-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
])

ENGLISH_FONT_PATH = find_font([
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
])

if not HINDI_FONT_PATH:
    HINDI_FONT_PATH = ENGLISH_FONT_PATH

if not ENGLISH_FONT_PATH:
    ENGLISH_FONT_PATH = HINDI_FONT_PATH


def load_font(size):

    return ImageFont.truetype(
        HINDI_FONT_PATH,
        size
    )


def load_english_font(size):

    return ImageFont.truetype(
        ENGLISH_FONT_PATH,
        size
    )


# ============================================================
# LOAD CHARACTER IMAGES
# ============================================================

def load_character_images():

    images = {}

    for pose, filename in CHARACTER_FILES.items():

        path = os.path.join(ASSET_DIR, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Character image not found: {path}"
            )

        image = Image.open(path).convert("RGBA")

        image.thumbnail(
            (CHARACTER_WIDTH, CHARACTER_HEIGHT),
            Image.Resampling.LANCZOS
        )

        images[pose] = image

    return images


# ============================================================
# AUDIO CONVERSION
# ============================================================

def convert_audio_to_wav(audio_path, wav_path):

    command = [
        "ffmpeg",
        "-y",
        "-i", audio_path,
        "-ac", "1",
        "-ar", "8000",
        "-f", "wav",
        wav_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Audio conversion failed: "
            + result.stderr.decode(
                "utf-8",
                errors="ignore"
            )[-2000:]
        )


# ============================================================
# AUDIO VOLUME ANALYSIS
# ============================================================

def analyse_audio(wav_path):

    audio = AudioFileClip(wav_path)

    duration = audio.duration

    audio_array = audio.to_soundarray(
        fps=8000
    )

    if audio_array.ndim > 1:
        audio_array = np.mean(
            audio_array,
            axis=1
        )

    audio_array = audio_array.astype(
        np.float32
    )

    audio.close()

    sample_rate = 8000

    window_size = int(
        sample_rate * 0.10
    )

    rms_values = []

    for start in range(
        0,
        len(audio_array),
        window_size
    ):

        chunk = audio_array[
            start:start + window_size
        ]

        if len(chunk) == 0:
            continue

        rms = np.sqrt(
            np.mean(chunk ** 2)
        )

        rms_values.append(rms)

    rms_values = np.array(
        rms_values,
        dtype=np.float32
    )

    if len(rms_values) == 0:
        return duration, np.zeros(1)

    # Smooth sudden volume changes
    smoothing_window = min(
        5,
        len(rms_values)
    )

    kernel = np.ones(
        smoothing_window
    ) / smoothing_window

    rms_values = np.convolve(
        rms_values,
        kernel,
        mode="same"
    )

    return duration, rms_values


# ============================================================
# CHARACTER ANIMATION TIMELINE
# ============================================================

def create_animation_timeline(rms_values):

    if len(rms_values) == 0:
        return ["neutral"]

    noise_floor = np.percentile(
        rms_values,
        25
    )

    speech_level = np.percentile(
        rms_values,
        80
    )

    threshold = max(
        noise_floor * 1.8,
        speech_level * 0.12,
        0.008
    )

    wide_threshold = max(
        threshold * 3.5,
        speech_level * 0.65
    )

    timeline = []

    current_pose = "neutral"

    candidate_pose = None
    candidate_count = 0

    minimum_samples = int(
        MIN_POSE_DURATION / ANIMATION_SAMPLE_RATE
    )

    for index, volume in enumerate(rms_values):

        if current_pose == "neutral":

            if volume > threshold * 1.15:
                desired_pose = "speaking"
            else:
                desired_pose = "neutral"

        else:

            if volume < threshold * 0.80:
                desired_pose = "neutral"

            elif volume > wide_threshold:
                desired_pose = "wide"

            else:
                desired_pose = "speaking"

        if desired_pose == candidate_pose:

            candidate_count += 1

        else:

            candidate_pose = desired_pose
            candidate_count = 1

        # Do not change the character too frequently.
        if (
            candidate_count >= minimum_samples
            and desired_pose != current_pose
        ):

            current_pose = desired_pose

        timeline.append(current_pose)

    return timeline


def get_character_pose(t, timeline):

    index = int(
        t / ANIMATION_SAMPLE_RATE
    )

    index = min(
        index,
        len(timeline) - 1
    )

    pose = timeline[index]

    # Occasional explanation gesture.
    # Only show it when the character is speaking.

    gesture_position = (
        t % GESTURE_INTERVAL
    )

    if (
        pose != "neutral"
        and 2.0 <= gesture_position <= 3.0
    ):

        return "gesture"

    return pose


# ============================================================
# SCRIPT TO SUBTITLE CHUNKS
# ============================================================

def create_subtitle_chunks(script):

    script = script.strip()

    if not script:
        return []

    # Split by Hindi and English punctuation.
    sentences = re.split(
        r'(?<=[।!?])\s+|(?<=[.!?])\s+',
        script
    )

    chunks = []

    MAX_WORDS = 7

    for sentence in sentences:

        words = sentence.strip().split()

        if not words:
            continue

        # Split longer sentences into smaller
        # meaningful subtitle blocks.

        for i in range(
            0,
            len(words),
            MAX_WORDS
        ):

            chunk = " ".join(
                words[i:i + MAX_WORDS]
            )

            if chunk:
                chunks.append(chunk)

    return chunks


# ============================================================
# ESTIMATE SUBTITLE TIMINGS
# ============================================================

def create_subtitle_timeline(chunks, duration):

    if not chunks:
        return []

    # Longer chunks receive more speaking time.
    weights = []

    for chunk in chunks:

        weight = len(
            re.sub(r'\s+', '', chunk)
        )

        weights.append(
            max(weight, 1)
        )

    total_weight = sum(weights)

    timeline = []

    current_time = 0.0

    for chunk, weight in zip(
        chunks,
        weights
    ):

        chunk_duration = (
            duration * weight / total_weight
        )

        timeline.append({
            "text": chunk,
            "start": current_time,
            "end": current_time + chunk_duration
        })

        current_time += chunk_duration

    return timeline


# ============================================================
# TEXT WRAPPING
# ============================================================

def wrap_text(
    draw,
    text,
    font,
    max_width
):

    words = text.split()

    lines = []
    current_line = ""

    for word in words:

        test_line = (
            current_line + " " + word
        ).strip()

        width = draw.textbbox(
            (0, 0),
            test_line,
            font=font
        )[2]

        if width <= max_width:

            current_line = test_line

        else:

            if current_line:
                lines.append(current_line)

            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


# ============================================================
# DRAW SUBTITLE BLOCKS
# ============================================================

def draw_subtitles(
    canvas,
    draw,
    t,
    subtitle_timeline
):

    visible_chunks = []

    current_index = -1

    for index, item in enumerate(
        subtitle_timeline
    ):

        if t >= item["start"]:

            current_index = index

        else:
            break

    if current_index < 0:
        return

    start_index = max(
        0,
        current_index - MAX_VISIBLE_CHUNKS + 1
    )

    visible_chunks = subtitle_timeline[
        start_index:current_index + 1
    ]

    # Compact subtitle area.
    # No giant white panel.

    font = load_font(42)

    line_height = 62

    padding_x = 38
    padding_y = 28

    card_width = PANEL_WIDTH

    y = PANEL_Y

    for index, item in enumerate(
        visible_chunks
    ):

        is_current = (
            item["start"] <= t
        )

        text = item["text"]

        lines = wrap_text(
            draw,
            text,
            font,
            card_width - padding_x * 2
        )

        # Limit each subtitle block to 2 lines.
        lines = lines[:2]

        card_height = (
            padding_y * 2
            + len(lines) * line_height
        )

        if is_current:

            background = (
                35, 35, 35
            )

            text_color = (
                255, 204, 0
            )

        else:

            background = (
                245, 245, 245
            )

            text_color = (
                150, 150, 150
            )

        draw.rounded_rectangle(
            (
                PANEL_X,
                y,
                PANEL_X + card_width,
                y + card_height
            ),
            radius=22,
            fill=background
        )

        text_y = y + padding_y

        for line in lines:

            draw.text(
                (
                    PANEL_X + padding_x,
                    text_y
                ),
                line,
                font=font,
                fill=text_color,
                stroke_width=0
            )

            text_y += line_height

        y += card_height + 18


# ============================================================
# CREATE VIDEO FRAME
# ============================================================

def create_frame(
    t,
    character_images,
    animation_timeline,
    subtitle_timeline
):

    canvas = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        BACKGROUND
    )

    draw = ImageDraw.Draw(canvas)

    # --------------------------------------------------------
    # CHARACTER
    # --------------------------------------------------------

    pose = get_character_pose(
        t,
        animation_timeline
    )

    character = character_images[
        pose
    ]

    x = CHARACTER_X

    y = (
        HEIGHT
        - character.height
        - 65
    )

    canvas.paste(
        character,
        (x, y),
        character
    )

    # --------------------------------------------------------
    # SUBTITLE BLOCKS
    # --------------------------------------------------------

    draw_subtitles(
        canvas,
        draw,
        t,
        subtitle_timeline
    )

    # --------------------------------------------------------
    # BRANDING
    # --------------------------------------------------------

    brand_font = load_english_font(28)

    draw.text(
        (75, 1010),
        "THE PTG GUY",
        font=brand_font,
        fill=(90, 90, 90)
    )

    # Small separator
    draw.line(
        (850, 930, 1820, 930),
        fill=(225, 225, 225),
        width=2
    )

    footer_font = load_font(27)

    draw.text(
        (900, 960),
        "Punjabi Lyrics Explained in Hindi",
        font=footer_font,
        fill=(110, 110, 110)
    )

    return np.array(canvas)


# ============================================================
# MAIN VIDEO GENERATION
# ============================================================

def generate_video(script, audio_base64):

    temp_dir = tempfile.mkdtemp()

    audio_path = os.path.join(
        temp_dir,
        "voice.mp3"
    )

    wav_path = os.path.join(
        temp_dir,
        "voice.wav"
    )

    video_path = os.path.join(
        temp_dir,
        "output.mp4"
    )

    try:

        # ----------------------------------------------------
        # DECODE AUDIO
        # ----------------------------------------------------

        audio_bytes = base64.b64decode(
            audio_base64
        )

        with open(
            audio_path,
            "wb"
        ) as audio_file:

            audio_file.write(
                audio_bytes
            )

        # ----------------------------------------------------
        # LOAD CHARACTER
        # ----------------------------------------------------

        character_images = (
            load_character_images()
        )

        # ----------------------------------------------------
        # AUDIO ANALYSIS
        # ----------------------------------------------------

        convert_audio_to_wav(
            audio_path,
            wav_path
        )

        duration, rms_values = (
            analyse_audio(wav_path)
        )

        if duration <= 0:
            raise RuntimeError(
                "Invalid audio duration."
            )

        # ----------------------------------------------------
        # CHARACTER ANIMATION
        # ----------------------------------------------------

        animation_timeline = (
            create_animation_timeline(
                rms_values
            )
        )

        # ----------------------------------------------------
        # SUBTITLE TIMINGS
        # ----------------------------------------------------

        chunks = create_subtitle_chunks(
            script
        )

        subtitle_timeline = (
            create_subtitle_timeline(
                chunks,
                duration
            )
        )

        # ----------------------------------------------------
        # GENERATE VIDEO
        # ----------------------------------------------------

        def make_frame(t):

            return create_frame(
                t,
                character_images,
                animation_timeline,
                subtitle_timeline
            )

        video_clip = VideoClip(
            make_frame,
            duration=duration
        )

        audio_clip = AudioFileClip(
            audio_path
        )

        # Compatibility with MoviePy versions.
        if hasattr(video_clip, "with_audio"):

            video_clip = video_clip.with_audio(
                audio_clip
            )

        else:

            video_clip = video_clip.set_audio(
                audio_clip
            )

        video_clip.write_videofile(
            video_path,
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            bitrate=VIDEO_BITRATE,
            preset="ultrafast",
            threads=4,
            logger=None
        )

        video_clip.close()
        audio_clip.close()

        # ----------------------------------------------------
        # ENCODE VIDEO
        # ----------------------------------------------------

        with open(
            video_path,
            "rb"
        ) as video_file:

            video_base64 = base64.b64encode(
                video_file.read()
            ).decode("utf-8")

        return {
            "video_base64": video_base64,
            "duration": round(duration, 2),
            "resolution": "1920x1080",
            "fps": FPS,
            "subtitle_chunks": len(chunks),
            "voice": "Sarvam AI"
        }

    finally:

        gc.collect()

        # Clean temporary files.
        import shutil

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(event):

    try:

        input_data = event.get(
            "input",
            {}
        )

        script = input_data.get(
            "script",
            ""
        ).strip()

        audio_base64 = input_data.get(
            "audio_base64",
            ""
        )

        if not script:
            return {
                "error": "Script is missing."
            }

        if not audio_base64:
            return {
                "error": "Audio is missing."
            }

        print(
            "Starting PTG video generation..."
        )

        print(
            f"Script length: {len(script)}"
        )

        result = generate_video(
            script,
            audio_base64
        )

        print(
            "Video generation completed successfully."
        )

        return result

    except Exception as error:

        import traceback

        traceback.print_exc()

        return {
            "error": str(error)
        }


runpod.serverless.start({
    "handler": handler
})
