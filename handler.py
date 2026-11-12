import os
import re
import gc
import glob
import time
import base64
import shutil
import tempfile
import traceback
import subprocess

import numpy as np
import runpod

from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoClip, AudioFileClip


# =====================================================
# CONFIGURATION
# =====================================================

WIDTH = 1920
HEIGHT = 1080
FPS = 30

ASSET_DIR = "/app"

CHARACTER_FILES = {
    "neutral": "01_neutral.png",
    "speaking": "02_mouth_slightly_open.png",
    "wide": "03_mouth_wide_open.png",
    "gesture": "09_both_hands_gesture.png"
}

CHARACTER_WIDTH = 760
CHARACTER_HEIGHT = 900

ANIMATION_INTERVAL = 0.10
MIN_POSE_DURATION = 0.65

PANEL_X = 850
PANEL_Y = 250
PANEL_WIDTH = 990

MAX_VISIBLE_CHUNKS = 4


# =====================================================
# FONT LOADING
# =====================================================

def find_hindi_font():

    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagariUI-Regular.ttf"
    ]

    for path in candidates:

        if os.path.exists(path):
            return path

    matches = glob.glob(
        "/usr/share/fonts/**/*Devanagari*Regular*",
        recursive=True
    )

    if matches:
        return matches[0]

    return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


HINDI_FONT_PATH = find_hindi_font()

ENGLISH_FONT_PATH = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
)


def get_font(size):

    return ImageFont.truetype(
        HINDI_FONT_PATH,
        size
    )


def get_english_font(size):

    return ImageFont.truetype(
        ENGLISH_FONT_PATH,
        size
    )


# =====================================================
# LOAD CHARACTER IMAGES
# =====================================================

def load_character_images():

    images = {}

    for pose, filename in CHARACTER_FILES.items():

        path = os.path.join(
            ASSET_DIR,
            filename
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Missing character image: {path}"
            )

        image = Image.open(
            path
        ).convert("RGBA")

        image.thumbnail(
            (CHARACTER_WIDTH, CHARACTER_HEIGHT),
            Image.Resampling.LANCZOS
        )

        images[pose] = image

    print(
        "All four character images loaded.",
        flush=True
    )

    return images


# =====================================================
# AUDIO ANALYSIS
# =====================================================

def analyse_audio(audio_path):

    raw_path = audio_path + ".raw"

    command = [
        "ffmpeg",
        "-y",
        "-i", audio_path,
        "-ac", "1",
        "-ar", "8000",
        "-f", "f32le",
        raw_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg audio analysis failed: "
            + result.stderr.decode(
                "utf-8",
                errors="ignore"
            )[-2000:]
        )

    samples = np.fromfile(
        raw_path,
        dtype=np.float32
    )

    os.remove(raw_path)

    if len(samples) == 0:

        raise RuntimeError(
            "Audio analysis returned empty data."
        )

    sample_rate = 8000

    window_size = int(
        sample_rate * ANIMATION_INTERVAL
    )

    rms_values = []

    for start in range(
        0,
        len(samples),
        window_size
    ):

        chunk = samples[
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

    if len(rms_values) >= 3:

        kernel = np.ones(3) / 3

        rms_values = np.convolve(
            rms_values,
            kernel,
            mode="same"
        )

    return rms_values


# =====================================================
# CHARACTER ANIMATION TIMELINE
# =====================================================

def create_animation_timeline(rms_values):

    if len(rms_values) == 0:
        return ["neutral"]

    noise = np.percentile(
        rms_values,
        25
    )

    speech = np.percentile(
        rms_values,
        80
    )

    threshold = max(
        noise * 1.8,
        speech * 0.12,
        0.008
    )

    wide_threshold = max(
        threshold * 3.5,
        speech * 0.65
    )

    timeline = []

    current_pose = "neutral"

    candidate_pose = None
    candidate_count = 0

    minimum_samples = max(
        1,
        int(
            MIN_POSE_DURATION / ANIMATION_INTERVAL
        )
    )

    for volume in rms_values:

        if volume > wide_threshold:

            desired_pose = "wide"

        elif volume > threshold:

            desired_pose = "speaking"

        else:

            desired_pose = "neutral"

        if desired_pose == candidate_pose:

            candidate_count += 1

        else:

            candidate_pose = desired_pose
            candidate_count = 1

        if (
            candidate_count >= minimum_samples
            and desired_pose != current_pose
        ):

            current_pose = desired_pose

        timeline.append(current_pose)

    return timeline


def get_character_pose(t, timeline):

    index = int(
        t / ANIMATION_INTERVAL
    )

    index = min(
        index,
        len(timeline) - 1
    )

    pose = timeline[index]

    gesture_position = t % 22.0

    if (
        pose != "neutral"
        and 8.0 <= gesture_position <= 9.0
    ):

        return "gesture"

    return pose


# =====================================================
# CREATE SUBTITLE CHUNKS
# =====================================================

def create_subtitle_chunks(script):

    sentences = re.split(
        r'(?<=[।!?])\s+|(?<=[.!?])\s+',
        script.strip()
    )

    chunks = []

    MAX_WORDS = 7

    for sentence in sentences:

        words = sentence.strip().split()

        if not words:
            continue

        for index in range(
            0,
            len(words),
            MAX_WORDS
        ):

            chunk = " ".join(
                words[index:index + MAX_WORDS]
            )

            if chunk:
                chunks.append(chunk)

    return chunks


# =====================================================
# SUBTITLE TIMING
# =====================================================

def create_subtitle_timeline(chunks, duration):

    if not chunks:
        return []

    weights = []

    for chunk in chunks:

        weights.append(
            max(len(chunk.split()), 1)
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


# =====================================================
# TEXT WRAPPING
# =====================================================

def wrap_text(draw, text, font, max_width):

    words = text.split()

    lines = []
    current = ""

    for word in words:

        test = (
            current + " " + word
        ).strip()

        width = draw.textbbox(
            (0, 0),
            test,
            font=font
        )[2]

        if width <= max_width:

            current = test

        else:

            if current:
                lines.append(current)

            current = word

    if current:
        lines.append(current)

    return lines


# =====================================================
# DRAW SUBTITLES
# =====================================================

def draw_subtitles(
    canvas,
    draw,
    t,
    subtitle_timeline
):

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

    visible = subtitle_timeline[
        start_index:current_index + 1
    ]

    font = get_font(42)

    line_height = 62

    padding_x = 35
    padding_y = 24

    y = PANEL_Y

    for item in visible:

        is_current = (
            item["start"] <= t
        )

        lines = wrap_text(
            draw,
            item["text"],
            font,
            PANEL_WIDTH - padding_x * 2
        )

        lines = lines[:2]

        card_height = (
            padding_y * 2
            + len(lines) * line_height
        )

        if is_current:

            background = (35, 35, 35)
            text_color = (255, 204, 0)

        else:

            background = (245, 245, 245)
            text_color = (145, 145, 145)

        draw.rounded_rectangle(
            (
                PANEL_X,
                y,
                PANEL_X + PANEL_WIDTH,
                y + card_height
            ),
            radius=20,
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
                fill=text_color
            )

            text_y += line_height

        y += card_height + 14


# =====================================================
# CREATE VIDEO FRAME
# =====================================================

def create_frame(
    t,
    character_images,
    animation_timeline,
    subtitle_timeline
):

    canvas = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        (255, 255, 255)
    )

    draw = ImageDraw.Draw(canvas)

    # CHARACTER

    pose = get_character_pose(
        t,
        animation_timeline
    )

    character = character_images[pose]

    character_x = 40

    character_y = (
        HEIGHT - character.height - 65
    )

    canvas.paste(
        character,
        (character_x, character_y),
        character
    )

    # SUBTITLES

    draw_subtitles(
        canvas,
        draw,
        t,
        subtitle_timeline
    )

    # BRANDING

    brand_font = get_english_font(28)

    draw.text(
        (75, 1010),
        "THE PTG GUY",
        font=brand_font,
        fill=(90, 90, 90)
    )

    footer_font = get_font(27)

    draw.text(
        (900, 960),
        "Punjabi Lyrics Explained in Hindi",
        font=footer_font,
        fill=(110, 110, 110)
    )

    return np.array(canvas)


# =====================================================
# GENERATE COMPLETE VIDEO
# =====================================================

def generate_video(script, audio_base64):

    temp_dir = tempfile.mkdtemp(
        prefix="ptg_"
    )

    audio_path = os.path.join(
        temp_dir,
        "sarvam_voice.wav"
    )

    video_path = os.path.join(
        temp_dir,
        "ptg_video.mp4"
    )

    audio_clip = None
    video_clip = None

    try:

        # STEP 1: RECEIVE AUDIO FROM FRONTEND

        print(
            "STEP 1: Receiving Sarvam audio from frontend...",
            flush=True
        )

        try:
            audio_bytes = base64.b64decode(
                audio_base64,
                validate=True
            )
        except Exception as error:
            raise ValueError(
                f"Invalid base64 audio received: {error}"
            )

        if not audio_bytes:

            raise RuntimeError(
                "Received empty audio from frontend."
            )

        with open(
            audio_path,
            "wb"
        ) as file:

            file.write(audio_bytes)

        print(
            f"Audio received: {len(audio_bytes)} bytes",
            flush=True
        )

        # STEP 2: LOAD AUDIO

        print(
            "STEP 2: Loading audio...",
            flush=True
        )

        audio_clip = AudioFileClip(
            audio_path
        )

        duration = audio_clip.duration

        if duration <= 0:

            raise RuntimeError(
                "Invalid audio duration."
            )

        # STEP 3: LOAD CHARACTER

        print(
            "STEP 3: Loading character images...",
            flush=True
        )

        character_images = (
            load_character_images()
        )

        # STEP 4: ANALYSE AUDIO

        print(
            "STEP 4: Analysing audio...",
            flush=True
        )

        rms_values = analyse_audio(
            audio_path
        )

        animation_timeline = (
            create_animation_timeline(
                rms_values
            )
        )

        # STEP 5: PREPARE SUBTITLES

        print(
            "STEP 5: Preparing subtitles...",
            flush=True
        )

        chunks = create_subtitle_chunks(
            script
        )

        subtitle_timeline = (
            create_subtitle_timeline(
                chunks,
                duration
            )
        )

        # STEP 6: GENERATE VIDEO FRAMES

        print(
            "STEP 6: Rendering video...",
            flush=True
        )

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

        # ATTACH THE SAME FRONTEND AUDIO

        video_clip = video_clip.set_audio(
            audio_clip
        )

        # STEP 7: EXPORT FINAL MP4

        print(
            "STEP 7: Exporting final MP4...",
            flush=True
        )

        video_clip.write_videofile(
            video_path,
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            bitrate="4000k",
            preset="ultrafast",
            threads=2,
            logger=None
        )

        video_clip.close()
        video_clip = None

        audio_clip.close()
        audio_clip = None

        # STEP 8: RETURN VIDEO

        print(
            "STEP 8: Encoding final video...",
            flush=True
        )

        with open(
            video_path,
            "rb"
        ) as file:

            video_base64 = base64.b64encode(
                file.read()
            ).decode("utf-8")

        print(
            "PTG VIDEO GENERATED SUCCESSFULLY.",
            flush=True
        )

        return {
            "video_base64": video_base64,
            "duration": round(duration, 2),
            "resolution": "1920x1080",
            "fps": FPS,
            "voice": "Sarvam Bulbul v3 - Shubh",
            "subtitle_chunks": len(chunks)
        }

    finally:

        if video_clip is not None:
            video_clip.close()

        if audio_clip is not None:
            audio_clip.close()

        gc.collect()

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# =====================================================
# RUNPOD HANDLER
# =====================================================

def handler(event):

    print(
        "PTG WORKER RECEIVED REQUEST.",
        flush=True
    )

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

            raise ValueError(
                "Hindi script is missing."
            )

        if not audio_base64:

            raise ValueError(
                "Audio is missing from frontend request."
            )

        if len(script) > 2500:

            raise ValueError(
                "Script exceeds 2500 characters."
            )

        print(
            f"Script received: {len(script)} characters",
            flush=True
        )

        result = generate_video(
            script,
            audio_base64
        )

        return result

    except Exception as error:

        print(
            "PTG WORKER ERROR:",
            str(error),
            flush=True
        )

        traceback.print_exc()

        return {
            "error": str(error)
        }


# =====================================================
# START RUNPOD
# =====================================================

print(
    "PTG WORKER STARTING...",
    flush=True
)

runpod.serverless.start({
    "handler": handler
})
