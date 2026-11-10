import os
import base64
import logging
import tempfile
import subprocess

import runpod
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageOps

from moviepy.editor import AudioFileClip, VideoClip


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =====================================================
# VIDEO CONFIGURATION
# =====================================================

WIDTH = 1920
HEIGHT = 1080

FPS = 30

VIDEO_BITRATE = "4000k"

ASSET_DIR = "/app"

AVATAR_WIDTH = 680
AVATAR_HEIGHT = 900

AUDIO_SAMPLE_RATE = 8000
AUDIO_CHUNK_SECONDS = 0.05


# =====================================================
# CHARACTER IMAGE FILES
# =====================================================

CHARACTER_FILES = {
    "neutral": "01_neutral.png",
    "mouth_slight": "02_mouth_slightly_open.png",
    "mouth_wide": "03_mouth_wide_open.png",
    "gesture": "09_both_hands_gesture.png"
}


# =====================================================
# LOAD CHARACTER IMAGES
# =====================================================

def load_character_images():

    logging.info("Loading character images...")

    characters = {}

    for pose, filename in CHARACTER_FILES.items():

        image_path = os.path.join(
            ASSET_DIR,
            filename
        )

        if not os.path.exists(image_path):

            raise FileNotFoundError(
                f"Character image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = ImageOps.contain(
            image,
            (AVATAR_WIDTH, AVATAR_HEIGHT),
            Image.Resampling.LANCZOS
        )

        characters[pose] = image

        logging.info(
            f"Loaded character: {pose}"
        )

    logging.info(
        "All four character images loaded successfully."
    )

    return characters


CHARACTERS = load_character_images()


# =====================================================
# LOAD HINDI FONTS
# =====================================================

def load_hindi_font(size):

    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagariUI-Regular.ttf"
    ]

    for path in font_paths:

        if os.path.exists(path):

            return ImageFont.truetype(
                path,
                size
            )

    logging.warning(
        "Hindi font not found. Using default font."
    )

    return ImageFont.load_default()


TITLE_FONT = load_hindi_font(48)
SCRIPT_FONT = load_hindi_font(46)
SMALL_FONT = load_hindi_font(25)


# =====================================================
# CONVERT MP3 TO WAV
# =====================================================

def convert_audio_to_wav(
    mp3_path,
    wav_path
):

    logging.info(
        "Converting MP3 audio to WAV..."
    )

    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        mp3_path,
        "-ar",
        "44100",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        wav_path
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    if not os.path.exists(wav_path):

        raise RuntimeError(
            "Audio conversion failed."
        )

    logging.info(
        "Audio converted successfully."
    )


# =====================================================
# ANALYSE AUDIO USING FFMPEG
# =====================================================

def analyze_audio(audio_path):

    logging.info(
        "Starting audio analysis..."
    )

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        audio_path,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "pipe:1"
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )

    audio_array = np.frombuffer(
        result.stdout,
        dtype=np.float32
    )

    chunk_size = int(
        AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SECONDS
    )

    rms_values = []

    for start in range(
        0,
        len(audio_array),
        chunk_size
    ):

        chunk = audio_array[
            start:start + chunk_size
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

        return np.zeros(1), 0.003

    # Smooth audio volume to avoid rapid switching.

    smoothing_window = np.ones(5) / 5

    rms_values = np.convolve(
        rms_values,
        smoothing_window,
        mode="same"
    )

    # Calculate adaptive silence threshold.

    noise_level = np.percentile(
        rms_values,
        20
    )

    maximum_volume = np.max(
        rms_values
    )

    threshold = max(
        noise_level * 1.5,
        maximum_volume * 0.055,
        0.003
    )

    logging.info(
        f"Audio analysis completed. Threshold: {threshold:.5f}"
    )

    return rms_values, threshold


# =====================================================
# GET AUDIO VOLUME
# =====================================================

def get_audio_volume(
    current_time,
    rms_values
):

    index = int(
        current_time / AUDIO_CHUNK_SECONDS
    )

    index = max(
        0,
        min(
            index,
            len(rms_values) - 1
        )
    )

    return float(
        rms_values[index]
    )


# =====================================================
# CALCULATE WORD TIMINGS
# =====================================================

def calculate_word_timings(
    script,
    duration
):

    words = script.split()

    if not words:
        return []

    weights = [
        max(len(word), 1)
        for word in words
    ]

    total_weight = sum(weights)

    timings = []

    current_time = 0.0

    for index, word in enumerate(words):

        word_duration = (
            duration *
            weights[index] /
            total_weight
        )

        start_time = current_time

        end_time = (
            current_time +
            word_duration
        )

        timings.append({
            "word": word,
            "start": start_time,
            "end": end_time
        })

        current_time = end_time

    return timings


# =====================================================
# GET CURRENT WORD
# =====================================================

def get_current_word(
    current_time,
    timings
):

    for index, item in enumerate(timings):

        if (
            item["start"]
            <= current_time
            < item["end"]
        ):

            return index

    if timings and current_time >= timings[-1]["end"]:

        return len(timings) - 1

    return 0


# =====================================================
# CHARACTER ANIMATION LOGIC
# =====================================================

def select_character_pose(
    current_time,
    volume,
    threshold
):

    # Silence detection

    if volume < threshold:

        return "neutral"

    # Occasional hand gesture during speaking.
    # This is a timed visual effect, not semantic detection.

    gesture_position = current_time % 16

    if 8.0 <= gesture_position < 8.8:

        return "gesture"

    # Stronger audio volume = wider mouth.

    if volume > threshold * 2.2:

        return "mouth_wide"

    # Normal speech.

    return "mouth_slight"


# =====================================================
# RENDER HINDI LYRICS PANEL
# =====================================================

def render_lyrics_panel(
    script,
    current_word_index
):

    panel = Image.new(
        "RGB",
        (1080, 820),
        (255, 255, 255)
    )

    draw = ImageDraw.Draw(panel)

    # Panel background

    draw.rounded_rectangle(
        (5, 5, 1075, 815),
        radius=30,
        fill=(255, 255, 255),
        outline=(225, 225, 225),
        width=3
    )

    # Heading

    draw.text(
        (45, 35),
        "गीत का मतलब",
        font=TITLE_FONT,
        fill=(25, 25, 25)
    )

    draw.line(
        (45, 110, 1035, 110),
        fill=(220, 220, 220),
        width=3
    )

    words = script.split()

    # Display a moving window of words.

    window_size = 18

    start_index = max(
        0,
        current_word_index - 8
    )

    if start_index + window_size > len(words):

        start_index = max(
            0,
            len(words) - window_size
        )

    visible_words = words[
        start_index:start_index + window_size
    ]

    x = 45
    y = 160

    max_width = 970

    line_height = 85

    space_width = draw.textlength(
        " ",
        font=SCRIPT_FONT
    )

    for offset, word in enumerate(visible_words):

        actual_index = (
            start_index + offset
        )

        word_width = draw.textlength(
            word,
            font=SCRIPT_FONT
        )

        # Move to next line if required.

        if (
            x > 45
            and x + word_width > 45 + max_width
        ):

            x = 45
            y += line_height

        # Highlight current word.

        if actual_index == current_word_index:

            draw.rounded_rectangle(
                (
                    x - 8,
                    y - 5,
                    x + word_width + 12,
                    y + 65
                ),
                radius=12,
                fill=(255, 225, 110)
            )

            text_color = (25, 25, 25)

        elif actual_index < current_word_index:

            text_color = (75, 75, 75)

        else:

            text_color = (155, 155, 155)

        draw.text(
            (x, y),
            word,
            font=SCRIPT_FONT,
            fill=text_color
        )

        x += (
            word_width +
            space_width
        )

    return panel


# =====================================================
# CREATE VIDEO FRAME
# =====================================================

def create_video_frame(
    current_time,
    timings,
    rms_values,
    threshold,
    script,
    panel_cache
):

    frame = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        (255, 255, 255)
    )

    draw = ImageDraw.Draw(frame)

    # Get current audio volume.

    volume = get_audio_volume(
        current_time,
        rms_values
    )

    # Find current word.

    current_index = get_current_word(
        current_time,
        timings
    )

    # Select character pose.

    pose = select_character_pose(
        current_time,
        volume,
        threshold
    )

    character = CHARACTERS[pose]

    # Character position.

    character_x = 45
    character_y = 90

    frame.paste(
        character,
        (character_x, character_y)
    )

    # Character branding.

    draw.text(
        (80, 1010),
        "THE PTG GUY",
        font=SMALL_FONT,
        fill=(75, 75, 75)
    )

    # Cache lyrics panel by current word.

    if current_index not in panel_cache:

        panel_cache[current_index] = render_lyrics_panel(
            script,
            current_index
        )

    panel = panel_cache[current_index]

    # Display lyrics panel.

    frame.paste(
        panel,
        (790, 130)
    )

    # Bottom branding.

    draw.text(
        (800, 1000),
        "Punjabi Lyrics Explained in Hindi",
        font=SMALL_FONT,
        fill=(110, 110, 110)
    )

    return np.array(frame)


# =====================================================
# GENERATE VIDEO
# =====================================================

def generate_video(
    script,
    audio_base64
):

    logging.info(
        "Starting video generation..."
    )

    with tempfile.TemporaryDirectory() as temp_dir:

        mp3_path = os.path.join(
            temp_dir,
            "narration.mp3"
        )

        wav_path = os.path.join(
            temp_dir,
            "narration.wav"
        )

        output_path = os.path.join(
            temp_dir,
            "output.mp4"
        )

        # Decode audio.

        try:

            audio_bytes = base64.b64decode(
                audio_base64,
                validate=True
            )

        except Exception as e:

            raise ValueError(
                f"Invalid base64 audio: {str(e)}"
            )

        with open(
            mp3_path,
            "wb"
        ) as audio_file:

            audio_file.write(
                audio_bytes
            )

        logging.info(
            "Audio file saved."
        )

        # Convert MP3 to WAV.

        convert_audio_to_wav(
            mp3_path,
            wav_path
        )

        # Analyse WAV.

        rms_values, threshold = analyze_audio(
            wav_path
        )

        audio_clip = None
        video_clip = None
        final_video = None

        try:

            # Open WAV using MoviePy.

            audio_clip = AudioFileClip(
                wav_path
            )

            duration = audio_clip.duration

            logging.info(
                f"Audio duration: {duration:.2f} seconds"
            )

            if duration <= 0:

                raise ValueError(
                    "Audio duration is invalid."
                )

            # Calculate estimated word timings.

            timings = calculate_word_timings(
                script,
                duration
            )

            # Cache rendered text panels.

            panel_cache = {}

            # Frame generator.

            def make_frame(t):

                return create_video_frame(
                    current_time=t,
                    timings=timings,
                    rms_values=rms_values,
                    threshold=threshold,
                    script=script,
                    panel_cache=panel_cache
                )

            video_clip = VideoClip(
                make_frame=make_frame,
                duration=duration
            )

            video_clip = video_clip.set_fps(
                FPS
            )

            final_video = video_clip.set_audio(
                audio_clip
            )

            logging.info(
                "Starting video export..."
            )

            final_video.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=FPS,
                preset="ultrafast",
                bitrate=VIDEO_BITRATE,
                threads=2,
                logger=None
            )

            logging.info(
                "Video export completed."
            )

        finally:

            # Close clips before temporary files are removed.

            if final_video is not None:
                final_video.close()

            if video_clip is not None:
                video_clip.close()

            if audio_clip is not None:
                audio_clip.close()

        # Read generated video.

        with open(
            output_path,
            "rb"
        ) as video_file:

            video_bytes = video_file.read()

        video_base64 = base64.b64encode(
            video_bytes
        ).decode("utf-8")

        logging.info(
            "Video generation completed successfully."
        )

        return {
            "video_base64": video_base64,
            "duration": round(duration, 2),
            "resolution": "1920x1080",
            "fps": FPS,
            "voice": "Sarvam Bulbul V3 - Shubh"
        }


# =====================================================
# RUNPOD SERVERLESS HANDLER
# =====================================================

def handler(job):

    logging.info(
        "Received new RunPod job."
    )

    job_input = job.get(
        "input",
        {}
    )

    script = job_input.get(
        "script"
    )

    audio_base64 = job_input.get(
        "audio_base64"
    )

    if not script:

        return {
            "error": "Hindi script is missing."
        }

    if not audio_base64:

        return {
            "error": "Audio data is missing."
        }

    try:

        return generate_video(
            script,
            audio_base64
        )

    except Exception as e:

        logging.exception(
            "Video generation failed."
        )

        return {
            "error": str(e)
        }


# =====================================================
# START RUNPOD WORKER
# =====================================================

if __name__ == "__main__":

    logging.info(
        "Starting PTG Studio RunPod worker..."
    )

    runpod.serverless.start({
        "handler": handler
    })
