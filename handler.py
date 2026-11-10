import os
import base64
import logging
import tempfile
import subprocess

import runpod
import numpy as np

from PIL import Image, ImageDraw, ImageFont

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

# Animation settings

POSE_ANALYSIS_INTERVAL = 0.10

MIN_POSE_DURATION = 0.55

VOLUME_SMOOTHING_SECONDS = 0.30

WIDE_MOUTH_MULTIPLIER = 3.5

# Gesture settings

GESTURE_INTERVAL = 22.0

GESTURE_DURATION = 1.0


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
# FONT LOADING
# =====================================================

def find_font(font_names):

    search_directories = [

        "/usr/share/fonts/truetype/noto",

        "/usr/share/fonts/truetype/dejavu",

        "/usr/share/fonts/truetype/liberation",

        "/usr/share/fonts"

    ]

    for directory in search_directories:

        for font_name in font_names:

            path = os.path.join(
                directory,
                font_name
            )

            if os.path.exists(path):

                return path

    return None


def load_hindi_font(size):

    font_path = find_font([

        "NotoSansDevanagari-Regular.ttf",

        "NotoSansDevanagariUI-Regular.ttf"

    ])

    if font_path:

        return ImageFont.truetype(
            font_path,
            size
        )

    raise FileNotFoundError(
        "Hindi font not found."
    )


def load_english_font(size):

    font_path = find_font([

        "DejaVuSans.ttf",

        "LiberationSans-Regular.ttf"

    ])

    if font_path:

        return ImageFont.truetype(
            font_path,
            size
        )

    return ImageFont.load_default()


TITLE_FONT = load_hindi_font(42)

SCRIPT_FONT = load_hindi_font(43)

BRANDING_FONT = load_english_font(25)


# =====================================================
# LOAD CHARACTER IMAGES
# =====================================================

def load_character_images():

    logging.info(
        "Loading character images..."
    )

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

        # Maintain consistent dimensions.

        image.thumbnail(
            (
                AVATAR_WIDTH,
                AVATAR_HEIGHT
            ),
            Image.Resampling.LANCZOS
        )

        canvas = Image.new(
            "RGB",
            (
                AVATAR_WIDTH,
                AVATAR_HEIGHT
            ),
            (255, 255, 255)
        )

        x = (
            AVATAR_WIDTH - image.width
        ) // 2

        y = (
            AVATAR_HEIGHT - image.height
        )

        canvas.paste(
            image,
            (x, y)
        )

        characters[pose] = canvas

        logging.info(
            f"Loaded character: {pose}"
        )

    logging.info(
        "All character images loaded."
    )

    return characters


CHARACTERS = load_character_images()


# =====================================================
# CONVERT MP3 TO WAV
# =====================================================

def convert_audio_to_wav(
    mp3_path,
    wav_path
):

    logging.info(
        "Converting MP3 to WAV..."
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

    logging.info(
        "Audio conversion completed."
    )


# =====================================================
# ANALYSE AUDIO
# =====================================================

def analyze_audio(audio_path):

    logging.info(
        "Analyzing audio..."
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
        AUDIO_SAMPLE_RATE *
        AUDIO_CHUNK_SECONDS
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

    # Smooth short audio fluctuations.

    smoothing_window = np.ones(7) / 7

    rms_values = np.convolve(
        rms_values,
        smoothing_window,
        mode="same"
    )

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
        f"Audio threshold: {threshold:.5f}"
    )

    return rms_values, threshold


# =====================================================
# SMOOTH AUDIO VOLUME
# =====================================================

def smooth_audio_volume(rms_values):

    window_size = max(
        1,
        int(
            VOLUME_SMOOTHING_SECONDS /
            AUDIO_CHUNK_SECONDS
        )
    )

    window = np.ones(
        window_size,
        dtype=np.float32
    ) / window_size

    return np.convolve(
        rms_values,
        window,
        mode="same"
    )


# =====================================================
# GET AUDIO VOLUME
# =====================================================

def get_audio_volume(
    current_time,
    rms_values
):

    index = int(
        current_time /
        AUDIO_CHUNK_SECONDS
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

    weights = []

    for word in words:

        weight = max(
            len(word),
            1
        )

        # Slightly account for punctuation pauses.

        if word.endswith((".", "!", "?", "।")):

            weight += 3

        elif word.endswith((",", ";")):

            weight += 1

        weights.append(weight)

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

    if not timings:

        return 0

    for index, item in enumerate(timings):

        if (

            item["start"]
            <= current_time
            < item["end"]

        ):

            return index

    if current_time >= timings[-1]["end"]:

        return len(timings) - 1

    return 0


# =====================================================
# BUILD CHARACTER ANIMATION TIMELINE
# =====================================================

def build_pose_timeline(
    rms_values,
    threshold,
    duration
):

    logging.info(
        "Building character animation timeline..."
    )

    smoothed_values = smooth_audio_volume(
        rms_values
    )

    interval = POSE_ANALYSIS_INTERVAL

    timeline = []

    current_pose = "neutral"

    pose_start_time = 0.0

    candidate_pose = current_pose

    candidate_start_time = 0.0

    time_points = np.arange(
        0,
        duration,
        interval
    )

    for current_time in time_points:

        volume = get_audio_volume(
            current_time,
            smoothed_values
        )

        # Hysteresis reduces flickering around
        # the silence threshold.

        if current_pose == "neutral":

            speaking = (
                volume > threshold * 1.15
            )

        else:

            speaking = (
                volume > threshold * 0.80
            )

        if not speaking:

            desired_pose = "neutral"

        elif volume > threshold * WIDE_MOUTH_MULTIPLIER:

            desired_pose = "mouth_wide"

        else:

            desired_pose = "mouth_slight"

        # Candidate pose must remain stable before
        # we consider switching.

        if desired_pose != candidate_pose:

            candidate_pose = desired_pose

            candidate_start_time = current_time

        candidate_duration = (
            current_time -
            candidate_start_time
        )

        current_pose_duration = (
            current_time -
            pose_start_time
        )

        if (

            candidate_pose != current_pose

            and candidate_duration >= 0.20

            and current_pose_duration >= MIN_POSE_DURATION

        ):

            current_pose = candidate_pose

            pose_start_time = current_time

        # Occasional hand gestures during speech.

        gesture_position = (
            current_time % GESTURE_INTERVAL
        )

        if (

            speaking

            and 8.0 <= gesture_position
            < 8.0 + GESTURE_DURATION

        ):

            timeline.append(
                "gesture"
            )

        else:

            timeline.append(
                current_pose
            )

    logging.info(
        "Character animation timeline completed."
    )

    return timeline


# =====================================================
# GET CHARACTER POSE
# =====================================================

def get_character_pose(
    current_time,
    pose_timeline
):

    index = int(
        current_time /
        POSE_ANALYSIS_INTERVAL
    )

    index = max(
        0,
        min(
            index,
            len(pose_timeline) - 1
        )
    )

    return pose_timeline[index]


# =====================================================
# WRAP SUBTITLE WORDS
# =====================================================

def wrap_subtitle_words(
    words,
    draw,
    max_width
):

    lines = []

    current_line = []

    current_width = 0

    space_width = draw.textlength(
        " ",
        font=SCRIPT_FONT
    )

    for item in words:

        word = item["word"]

        word_width = draw.textlength(
            word,
            font=SCRIPT_FONT
        )

        required_width = word_width

        if current_line:

            required_width += space_width

        if (

            current_line

            and current_width + required_width > max_width

        ):

            lines.append(
                current_line
            )

            current_line = []

            current_width = 0

            required_width = word_width

        current_line.append(item)

        current_width += required_width

    if current_line:

        lines.append(
            current_line
        )

    return lines


# =====================================================
# RENDER COMPACT SUBTITLE PANEL
# =====================================================

def render_subtitle_panel(
    script,
    current_word_index
):

    panel = Image.new(
        "RGB",
        (1000, 500),
        (255, 255, 255)
    )

    draw = ImageDraw.Draw(panel)

    # Compact dark subtitle card.

    draw.rounded_rectangle(

        (10, 10, 990, 490),

        radius=28,

        fill=(32, 36, 45),

        outline=(65, 70, 80),

        width=2

    )

    # Heading.

    draw.text(

        (45, 35),

        "गीत का मतलब",

        font=TITLE_FONT,

        fill=(255, 255, 255)

    )

    # Divider.

    draw.line(

        (45, 105, 955, 105),

        fill=(85, 90, 100),

        width=2

    )

    words = script.split()

    # Show a smaller window of words.

    window_size = 12

    start_index = max(

        0,

        current_word_index - 5

    )

    if start_index + window_size > len(words):

        start_index = max(

            0,

            len(words) - window_size

        )

    visible_words = []

    for offset, word in enumerate(

        words[start_index:start_index + window_size]

    ):

        actual_index = (
            start_index + offset
        )

        visible_words.append({

            "word": word,

            "index": actual_index

        })

    lines = wrap_subtitle_words(

        visible_words,

        draw,

        890

    )

    # Compact three-line layout.

    y = 155

    line_height = 95

    for line in lines[:3]:

        x = 45

        space_width = draw.textlength(

            " ",

            font=SCRIPT_FONT

        )

        for item in line:

            word = item["word"]

            actual_index = item["index"]

            word_width = draw.textlength(

                word,

                font=SCRIPT_FONT

            )

            # Current spoken word.

            if actual_index == current_word_index:

                text_color = (
                    255,
                    210,
                    75
                )

                draw.rounded_rectangle(

                    (

                        x - 5,

                        y - 5,

                        x + word_width + 10,

                        y + 65

                    ),

                    radius=10,

                    fill=(85, 72, 35)

                )

            # Previously spoken words.

            elif actual_index < current_word_index:

                text_color = (
                    245,
                    245,
                    245
                )

            # Upcoming words.

            else:

                text_color = (
                    155,
                    160,
                    170
                )

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

        y += line_height

    return panel


# =====================================================
# CREATE VIDEO FRAME
# =====================================================

def create_video_frame(
    current_time,
    timings,
    pose_timeline,
    script,
    panel_cache
):

    frame = Image.new(

        "RGB",

        (WIDTH, HEIGHT),

        (255, 255, 255)

    )

    draw = ImageDraw.Draw(frame)

    # Current subtitle word.

    current_index = get_current_word(

        current_time,

        timings

    )

    # Current character pose.

    pose = get_character_pose(

        current_time,

        pose_timeline

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

        font=BRANDING_FONT,

        fill=(75, 75, 75)

    )

    # Subtitle panel caching.

    if current_index not in panel_cache:

        panel_cache[current_index] = render_subtitle_panel(

            script,

            current_index

        )

    panel = panel_cache[current_index]

    # Compact subtitle panel on right.

    frame.paste(

        panel,

        (850, 290)

    )

    # Bottom branding.

    draw.text(

        (870, 850),

        "Punjabi Lyrics Explained in Hindi",

        font=BRANDING_FONT,

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
            "Audio saved."
        )

        # Convert MP3 to WAV.

        convert_audio_to_wav(

            mp3_path,

            wav_path

        )

        # Analyze audio.

        rms_values, threshold = analyze_audio(

            wav_path

        )

        audio_clip = None

        video_clip = None

        final_video = None

        try:

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

            # Build smooth character animation.

            pose_timeline = build_pose_timeline(

                rms_values,

                threshold,

                duration

            )

            # Cache subtitle panels.

            panel_cache = {}

            # Frame generator.

            def make_frame(t):

                return create_video_frame(

                    current_time=t,

                    timings=timings,

                    pose_timeline=pose_timeline,

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
