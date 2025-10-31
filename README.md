Emoji Detector

A real-time camera-based emoji display app that uses MediaPipe to detect your face expressions and hand poses, then reacts with fun custom emojis — including freaky and monkey faces 🐵😜

Features

🧠 Face Detection: Detects your facial expressions and displays matching emojis

✋ Hand Detection: Recognizes raised hands using MediaPipe Pose

😜 Freaky Mode: Shows “freaky” emoji when certain gestures are detected

🐵 Monkey Thinking & Got-It Modes: Displays custom monkey emojis for unique reactions

⚡ Real-time Processing: Instant camera-to-emoji response

Requirements

Python 3.12 or later

macOS / Windows with webcam

See requirements.txt for dependencies

Installation
# Clone the repository
git clone https://github.com/<your-username>/emoji_detector.git
cd emoji_detector

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt


Place your emoji image files in the project folder:

freaky.jpeg – freaky face

monkeygotit.jpeg – monkey “got it” face

monkeythinking.jpg – thinking monkey

Usage
# Option A: Run using helper script
./run.sh

# Option B: Run manually
source venv/bin/activate
python3 emoji_detector.py


Two windows will appear:

🎥 Camera Feed: Live camera with detection outlines

😎 Emoji Output: Displays corresponding emoji

Controls:

Press q → Quit

Raise hands → Freaky emoji 😜

Smile / Think → Monkey emojis 🐵

How It Works

MediaPipe FaceMesh → Detects facial landmarks for expressions

MediaPipe Pose → Detects body joints and raised hands

OpenCV → Captures video and displays the emoji reaction window

Customization

🎚 Adjust thresholds — modify sensitivity in emoji_detector.py.
🖼 Replace emoji files — swap out images with your own.

Troubleshooting

Check Camera permissions in system privacy settings.

Make sure image filenames match exactly.

Ensure good lighting and full face visibility.

Dependencies

opencv-python

mediapipe

numpy

See requirements.txt and requirements_lock.txt for details.

License

MIT License – see LICENSE file for details.
