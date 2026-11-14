import os
import re
import gc
import glob
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
CHARACTER_X = 15
CHARACTER_Y = 90

ANIMATION_INTERVAL = 0.10
MIN_POSE_DURATION = 0.65

# Old subtitle panel layout
PANEL_X = 780
PANEL_WIDTH = WIDTH - PANEL_X
MAX_VISIBLE_SENTENCES = 4

SAMPLE_RATE = 8000


# =====================================================
# FONT LOADING
# =====================================================

def find_hindi_font():
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagariUI-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf"
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    matches = glob.glob(
        "/usr/share/fonts/**/*Devanagari*Regular*",
        recursive=True
    )

    for path in matches:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Hindi Devanagari font not found. "
        "Please ensure Noto Sans Devanagari is installed in the Docker image."
    )


def find_english_font():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    matches = glob.glob(
        "/usr/share/fonts/**/*Sans*Regular*.ttf",
        recursive=True
    )

    for path in matches:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError("English font not found.")


HINDI_FONT_PATH = find_hindi_font()
ENGLISH_FONT_PATH = find_english_font()


def get_font(size):
    return ImageFont.truetype(HINDI_FONT_PATH, size)


def get_english_font(size):
    return ImageFont.truetype(ENGLISH_FONT_PATH, size)


# =====================================================
# LOAD CHARACTER IMAGES
# =====================================================

def load_character_images():
    images = {}

    for pose, filename in CHARACTER_FILES.items():
        path = os.path.join(ASSET_DIR, filename)

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing character image: {path}"
            )

        image = Image.open(path).convert("RGBA")
        image.thumbnail(
            (CHARACTER_WIDTH, CHARACTER_HEIGHT),
            Image.Resampling.LANCZOS
        )
        images[pose] = image

    print("All four character images loaded.", flush=True)
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
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
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
            "FFmpeg audio analysis failed: " +
            result.stderr.decode(
                "utf-8",
                errors="ignore"
            )[-2000:]
        )

    try:
        samples = np.fromfile(raw_path, dtype=np.float32)
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)

    if len(samples) == 0:
        raise RuntimeError("Audio analysis returned empty data.")

    window_size = max(
        1,
        int(SAMPLE_RATE * ANIMATION_INTERVAL)
    )

    rms_values = []

    for start in range(0, len(samples), window_size):
        chunk = samples[start:start + window_size]

        if len(chunk) == 0:
            continue

        rms = np.sqrt(np.mean(chunk ** 2))
        rms_values.append(rms)

    rms_values = np.asarray(rms_values, dtype=np.float32)

    if len(rms_values) >= 3:
        rms_values = np.convolve(
            rms_values,
            np.ones(3, dtype=np.float32) / 3,
            mode="same"
        )

    return rms_values


# =====================================================
# CHARACTER ANIMATION TIMELINE
# =====================================================

def create_animation_timeline(rms_values):
    if len(rms_values) == 0:
        return ["neutral"]

    noise = float(np.percentile(rms_values, 25))
    speech = float(np.percentile(rms_values, 80))

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
        int(MIN_POSE_DURATION / ANIMATION_INTERVAL)
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
    if not timeline:
        return "neutral"

    index = int(t / ANIMATION_INTERVAL)
    index = max(0, min(index, len(timeline) - 1))

    pose = timeline[index]

    # Preserve the occasional gesture animation.
    gesture_position = t % 22.0

    if pose != "neutral" and 8.0 <= gesture_position <= 9.0:
        return "gesture"

    return pose


# =====================================================
# OLD SUBTITLE LOGIC: SENTENCE SPLITTING AND TIMING
# =====================================================

def split_script(script):
    lines = [
        line.strip()
        for line in script.splitlines()
        if line.strip()
    ]

    if len(lines) > 1:
        return lines

    sentences = re.split(
        r'(?<=[।.!?])\s+',
        script.strip()
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences if sentences else [script.strip()]


def calculate_sentence_timings(sentences, duration):
    if not sentences:
        return []

    total_chars = sum(
        len(sentence)
        for sentence in sentences
    )

    timings = []
    current_time = 0.0

    for sentence in sentences:
        if total_chars:
            sentence_duration = (
                len(sentence) / total_chars
            ) * duration
        else:
            sentence_duration = duration / len(sentences)

        timings.append({
            "text": sentence,
            "start": current_time,
            "end": current_time + sentence_duration
        })

        current_time += sentence_duration

    # Ensure the final sentence remains visible through the end.
    timings[-1]["end"] = duration

    return timings


# =====================================================
# SUBTITLE TEXT WRAPPING
# =====================================================

def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = (
            current_line + " " + word
        ).strip()

        box = draw.textbbox(
            (0, 0),
            test_line,
            font=font
        )

        text_width = box[2] - box[0]

        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)

            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


# =====================================================
# OLD SUBTITLE DESIGN DRAWN ON THE NEW VIDEO FRAME
# =====================================================

def draw_subtitles(canvas, draw, t, timings):
    # Old light-grey split-screen panel.
    draw.rectangle(
        (PANEL_X, 0, WIDTH, HEIGHT),
        fill=(250, 250, 250)
    )

    # Old panel heading.
    draw.text(
        (PANEL_X + 65, 75),
        "गीत का मतलब",
        font=get_font(55),
        fill=(25, 25, 25)
    )

    draw.rounded_rectangle(
        (
            PANEL_X + 65,
            155,
            WIDTH - 65,
            160
        ),
        radius=3,
        fill=(220, 220, 220)
    )

    if not timings:
        return

    current_index = 0

    for index, item in enumerate(timings):
        if item["start"] <= t < item["end"]:
            current_index = index
            break

        if t >= item["end"]:
            current_index = index

    start_index = max(
        0,
        current_index - MAX_VISIBLE_SENTENCES + 1
    )

    visible = timings[
        start_index:current_index + 1
    ]

    active_font = get_font(43)
    previous_font = get_font(34)

    y = 220
    card_left = PANEL_X + 65
    card_right = WIDTH - 65
    text_left = PANEL_X + 100
    max_text_width = PANEL_WIDTH - 190

    for local_index, item in enumerate(visible):
        actual_index = start_index + local_index
        is_current = actual_index == current_index

        if is_current:
            box_fill = (255, 238, 190)
            text_fill = (25, 25, 25)
            text_font = active_font
        else:
            box_fill = (240, 240, 240)
            text_fill = (125, 125, 125)
            text_font = previous_font

        wrapped = wrap_text(
            draw,
            item["text"],
            text_font,
            max_text_width
        )

        # Keep unusually long sentences inside the panel.
        available_height = HEIGHT - 70 - y
        max_lines = max(
            1,
            min(4, int((available_height - 45) / 65))
        )
        wrapped = wrapped[:max_lines]

        box_height = max(1, len(wrapped)) * 65 + 45

        draw.rounded_rectangle(
            (
                card_left,
                y,
                card_right,
                y + box_height
            ),
            radius=22,
            fill=box_fill
        )

        text_y = y + 20

        for line in wrapped:
            draw.text(
                (text_left, text_y),
                line,
                font=text_font,
                fill=text_fill
            )
            text_y += 65

        y += box_height + 25

        if y >= HEIGHT - 30:
            break


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

    # Character on the left.
    pose = get_character_pose(
        t,
        animation_timeline
    )

    character = character_images.get(
        pose,
        character_images["neutral"]
    )

    character_y = max(
        0,
        min(
            CHARACTER_Y,
            HEIGHT - character.height
        )
    )

    canvas.paste(
        character,
        (CHARACTER_X, character_y),
        character
    )

    # Old sentence-based subtitle panel on the right.
    draw_subtitles(
        canvas,
        draw,
        t,
        subtitle_timeline
    )

    # Existing PTG branding.
    draw.text(
        (75, 1010),
        "THE PTG GUY",
        font=get_english_font(28),
        fill=(90, 90, 90)
    )

    return np.array(canvas)


# =====================================================
# GENERATE COMPLETE VIDEO
# =====================================================

def generate_video(script, audio_base64):
    temp_dir = tempfile.mkdtemp(prefix="ptg_")

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
        # STEP 1: Decode audio received from frontend.
        print("STEP 1: Receiving Sarvam audio...", flush=True)

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

        with open(audio_path, "wb") as file:
            file.write(audio_bytes)

        print(
            f"Audio received: {len(audio_bytes)} bytes",
            flush=True
        )

        # STEP 2: Load audio.
        print("STEP 2: Loading audio...", flush=True)

        audio_clip = AudioFileClip(audio_path)
        duration = float(audio_clip.duration)

        if duration <= 0:
            raise RuntimeError("Invalid audio duration.")

        # STEP 3: Load the four character poses.
        print("STEP 3: Loading character images...", flush=True)

        character_images = load_character_images()

        # STEP 4: Analyse audio for mouth animation.
        print("STEP 4: Analysing audio...", flush=True)

        rms_values = analyse_audio(audio_path)

        animation_timeline = create_animation_timeline(
            rms_values
        )

        # STEP 5: Prepare old sentence-based subtitles.
        print("STEP 5: Preparing sentence subtitles...", flush=True)

        sentences = split_script(script)

        subtitle_timeline = calculate_sentence_timings(
            sentences,
            duration
        )

        # STEP 6: Render video frames.
        print("STEP 6: Rendering video...", flush=True)

        def make_frame(t):
            return create_frame(
                t,
                character_images,
                animation_timeline,
                subtitle_timeline
            )

        video_clip = VideoClip(
            make_frame=make_frame,
            duration=duration
        )

        video_clip = video_clip.set_audio(audio_clip)

        # STEP 7: Export MP4.
        print("STEP 7: Exporting final MP4...", flush=True)

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

        # STEP 8: Encode the final video.
        print("STEP 8: Encoding final video...", flush=True)

        if not os.path.isfile(video_path):
            raise RuntimeError("Video file was not created.")

        file_size = os.path.getsize(video_path)

        if file_size == 0:
            raise RuntimeError("Generated video is empty.")

        with open(video_path, "rb") as file:
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
            "subtitle_sentences": len(sentences),
            "layout": "Animated Character + Old Sentence Subtitle Panel",
            "file_size_mb": round(
                file_size / (1024 * 1024),
                2
            )
        }

    finally:
        if video_clip is not None:
            try:
                video_clip.close()
            except Exception:
                pass

        if audio_clip is not None:
            try:
                audio_clip.close()
            except Exception:
                pass

        gc.collect()

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# =====================================================
# RUNPOD SERVERLESS HANDLER
# =====================================================

def handler(event):
    print(
        "PTG WORKER RECEIVED REQUEST.",
        flush=True
    )

    try:
        input_data = event.get("input", {})

        script = input_data.get(
            "script",
            ""
        ).strip()

        audio_base64 = input_data.get(
            "audio_base64",
            ""
        )

        if not script:
            raise ValueError("Hindi script is missing.")

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

        return generate_video(
            script,
            audio_base64
        )

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

print("PTG WORKER STARTING...", flush=True)

runpod.serverless.start({
    "handler": handler
})
