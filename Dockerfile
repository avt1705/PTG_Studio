FROM python:3.10-slim

WORKDIR /app


# =====================================================
# INSTALL SYSTEM DEPENDENCIES
# =====================================================

RUN apt-get update && apt-get install -y \
    ffmpeg \
    fonts-noto-core \
    fonts-noto-extra \
    && rm -rf /var/lib/apt/lists/*


# =====================================================
# INSTALL PYTHON DEPENDENCIES
# =====================================================

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# =====================================================
# COPY CHARACTER IMAGES
# =====================================================

COPY 01_neutral.png /app/01_neutral.png

COPY 02_mouth_slightly_open.png /app/02_mouth_slightly_open.png

COPY 03_mouth_wide_open.png /app/03_mouth_wide_open.png

COPY 09_both_hands_gesture.png /app/09_both_hands_gesture.png


# =====================================================
# COPY HANDLER
# =====================================================

COPY handler.py /app/handler.py


# =====================================================
# START WORKER
# =====================================================

CMD ["python", "-u", "/app/handler.py"]
