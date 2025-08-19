# Audio-modulated-image-processing
Project containing some experiments on the implementation of an audio modulated image processing, using a pipewire/JACK inputs

# Requirements
- `python3`
- `pip3`
- `cv2`
- `numpy`
- `jack`

# Installation
```bash
python -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

# Run image fading with audio modulation and default parameters
```bash
python main.py --image <image_path>
```

# Run image fading + distortion with audio modulation and default parameters
```bash
python main-withDistortion.py --image <image_path>
```