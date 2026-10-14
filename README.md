<div align="center">

# FaceLens

**Face detection for webcams, videos and images, with a desktop app, emotion labels and one-click privacy blur.**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B%20(%3C5)-green)
![Detectors](https://img.shields.io/badge/detectors-YuNet%20%7C%20Haar-orange)
![Interface](https://img.shields.io/badge/interface-CLI%20%7C%20GUI-lightgrey)

[Features](#features) ·
[Quick start](#quick-start) ·
[CLI usage](#command-line-usage) ·
[Desktop app](#desktop-app-gui) ·
[Emotions](#emotion-recognition) ·
[Troubleshooting](#troubleshooting)

</div>

---

`facetool` finds faces in a live webcam feed, a video or a photo. Use it from the command line or from a point-and-click window. It draws boxes around faces and can blur them, mark landmarks, label expressions, save cropped faces and write the annotated result to disk.

It started as a 20-line Haar cascade webcam demo and grew into a small, usable tool with a modern detector, a desktop app and proper error handling.

## Features

| | |
|---|---|
| **Two detectors** | **YuNet** (default) is OpenCV's DNN detector. It handles angled and partly hidden faces and returns a confidence score plus five landmarks. **Haar cascade** is the classic lightweight option and needs no model download |
| **Any input** | Webcam, video file or image file, chosen automatically from the argument |
| **Desktop app** | Live preview, sliders and one-click saving in `facetool_gui.py` |
| **Emotion labels** | Optional. Tags each face as angry, disgust, fearful, happy, neutral, sad or surprised |
| **Privacy blur** | Blur every detected face with `--blur` |
| **Landmarks** | Draw eye, nose and mouth points with `--landmarks` |
| **Save results** | Annotated image or video (`-o`) and cropped faces (`--save-faces`) |
| **Live feedback** | Face count and FPS overlay, plus keyboard shortcuts |
| **Headless mode** | Run on a server with `--no-display` |
| **Robust** | Handles camera failures, always releases the camera, and gives clear errors instead of tracebacks |

## Quick start

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Then pick how you want to run it:

```bash
python facetool_gui.py     # desktop app
python facetool.py         # command line, opens your default webcam
```

Press `q` or `Esc` in the command-line window to quit. The YuNet model (about 230 KB) downloads automatically on first run into a `models/` folder next to the script.

> [!NOTE]
> `requirements.txt` pins OpenCV 4.x (`<5`) because OpenCV 5 removed the Haar `CascadeClassifier`. With OpenCV 5 the default YuNet detector works, but `--detector haar` and the automatic fallback do not.

## Command-line usage

```bash
python facetool.py                                    # webcam
python facetool.py photo.jpg                          # image -> photo_out.jpg
python facetool.py clip.mp4 -o out.mp4                # video -> annotated video
python facetool.py --blur --landmarks                 # webcam, blur faces and show landmarks
python facetool.py --emotions                         # webcam, label each face's expression
python facetool.py photo.jpg --emotions --landmarks   # image with emotion labels
python facetool.py --detector haar                    # use the classic Haar cascade
python facetool.py photo.jpg --save-faces crops       # also save cropped faces to ./crops
python facetool.py clip.mp4 --no-display -o out.mp4   # headless processing
python facetool.py 1                                  # second webcam
```

### Options

| Option | Description |
|---|---|
| `source` | Webcam index (default `0`), video file or image file |
| `-d`, `--detector` | `yunet` (default, more accurate) or `haar` (fast, basic) |
| `-t`, `--threshold` | YuNet confidence threshold, default `0.7`. Raise it for fewer false positives |
| `-o`, `--output` | Write the annotated image or video to this path |
| `--blur` | Blur detected faces |
| `--landmarks` | Draw eyes, nose and mouth (YuNet) or eyes (Haar) |
| `--emotions` | Label each face's expression (downloads a small model on first use) |
| `--save-faces DIR` | Save cropped faces and snapshots to `DIR` |
| `--no-display` | Do not open a window |
| `--max-frames N` | Stop after `N` frames (`0` means no limit) |

Run `python facetool.py --help` for the same list in your terminal.

### Live keys

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `s` | Save a snapshot of the current frame |
| `b` | Toggle face blur |
| `l` | Toggle landmarks |
| `e` | Toggle emotion labels (when started with `--emotions`) |

## Desktop app (GUI)

```bash
python facetool_gui.py
```

A window with a live preview and a control panel:

| Panel | What you can do |
|---|---|
| **Source** | Open a webcam (pick the camera number), an image or a video |
| **Detection** | Choose YuNet or Haar and adjust the confidence threshold with a slider |
| **Display** | Toggle face blur, landmarks and emotion labels. Changes apply instantly, even to a still image |
| **Save** | Snapshot of the current frame, cropped faces, or record the annotated stream to MP4 |
| **Output folder** | Defaults to `facetool_output/` next to where you launch it |

The GUI uses Tkinter and Pillow. Tkinter is bundled with Python on Windows and macOS. On Ubuntu/Debian install it with `sudo apt install python3-tk`. Recording works for webcams and videos, not for single images (use *Save snapshot* for those).

## Emotion recognition

Turn it on with `--emotions` (CLI) or the **Identify emotions** checkbox (GUI). Each face gets a colored box and one of seven labels: `angry`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised`.

It uses OpenCV's small facial-expression model (MobileFaceNet, trained on RAF-DB) from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition), run through OpenCV's built-in DNN module, so there are no extra dependencies. The model file downloads on first use into `models/`. If the download fails, detection keeps working without emotions. To use it offline, put `facial_expression_recognition_mobilefacenet_2022july.onnx` in `models/`.

> [!WARNING]
> Emotion labels describe the **visible expression**, not what someone feels, and they are experimental. Do not use them to make decisions about people (hiring, grading, security, health). See [Responsible use](#responsible-use).

**What to expect**

- A polite smile, a joke or a resting face can all be misread.
- Accuracy on a benchmark is not accuracy on your camera. Lighting, angle, glasses, masks, image quality and differences between people all matter, and results can be uneven across groups of people.
- It works best with **YuNet**, whose landmarks are used to straighten each face before classification. With Haar, faces are only cropped, so results are less reliable.
- Labels can flicker from frame to frame. Treat them as a demo, not a measurement.

## How it works

1. **Load** the chosen detector. YuNet falls back to Haar if the model cannot be loaded.
2. **Read** a frame from the webcam, video or image.
3. **Detect** faces and get each box, plus a score and landmarks where the detector provides them.
4. **Label** each face's expression, if emotions are on.
5. **Annotate** by optionally blurring faces, then drawing boxes, labels and landmarks, plus the face count and FPS overlay.
6. **Output** by showing the frame, writing it to a file, and saving crops if requested.

### Choosing a detector

| | YuNet (default) | Haar cascade |
|---|---|---|
| Accuracy | Better, handles angles and partial occlusion | Best on frontal, well-lit faces |
| Confidence score | Yes | No |
| Landmarks | Eyes, nose, mouth corners | Eyes only |
| Emotion accuracy | Better (faces are aligned first) | Lower (plain crop) |
| Extra download | Small model file on first run | None |
| Works with OpenCV 5 | Yes | No |

## Project structure

```
.
├── facetool.py         # command-line tool and detection/emotion engine
├── facetool_gui.py     # desktop app (Tkinter), reuses facetool.py
├── requirements.txt    # opencv-python (4.x), numpy, pillow
├── README.md
├── docs/               # single-page project site (GitHub Pages)
└── models/             # downloaded models, created on first use
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open source: 0` | No camera found or it is in use by another app. Try another index (`python facetool.py 1`) or close the other app |
| YuNet model fails to download | Download `face_detection_yunet_2023mar.onnx` from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and put it in a `models/` folder next to `facetool.py` |
| `no CascadeClassifier (removed in OpenCV 5)` | Install OpenCV 4.x: `pip install "opencv-python>=4.8,<5"` |
| Window does not open on a server | Use `--no-display` and write results with `-o` |
| `The GUI needs Tkinter and Pillow` | `pip install pillow`. On Ubuntu/Debian also `sudo apt install python3-tk` |
| `Emotion model unavailable` | Download `facial_expression_recognition_mobilefacenet_2022july.onnx` from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition) into `models/`, then enable the option again |
| Emotion labels look wrong or jumpy | Use YuNet (not Haar), face the camera with even lighting, and remember it only reads expressions. See [Emotion recognition](#emotion-recognition) |
| Too many false detections | Raise `--threshold` (for example `0.85`), or use YuNet instead of Haar |
| Faces missed | Lower `--threshold` (for example `0.5`) and check lighting |

## Project page

<details>
<summary>Publish the <code>docs/</code> site with GitHub Pages</summary>

The `docs/` folder is a single-page site that explains the tool: install, examples, options and the two detectors.

1. Push the repository to GitHub, including the `docs/` folder.
2. Open **Settings > Pages**.
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
4. Choose your main branch and the **/docs** folder, then save.
5. After a minute the page is live at `https://<your-username>.github.io/<repo-name>/`.

The page is plain HTML with no build step. To preview it locally, open `docs/index.html` in a browser.

</details>

## Ideas for next steps

- [ ] Smooth boxes and emotion labels across frames so they do not flicker
- [ ] Face recognition (who is this?) on top of detection
- [ ] A browser demo with Streamlit or Gradio

## Responsible use

Face detection involves people's images. Only run this on footage you have the right to process, follow the privacy laws that apply where you are, and tell people when a camera is analysing them. The `--blur` option is there to make privacy-friendly use easy.

Emotion labels deserve extra care. Facial expression is not a reliable window into someone's inner state, and models like this one make mistakes that fall unevenly across people. Use the feature for learning, demos and personal projects, not for judging, scoring or screening anyone.

## Credits

- [OpenCV](https://opencv.org/) for the vision library and the Haar cascades
- [YuNet](https://github.com/ShiqiYu/libfacedetection) by Shiqi Yu and contributors, distributed via the [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
- Facial-expression model: Progressive Teacher with a MobileFaceNet backbone, from the OpenCV Zoo (Apache 2.0)
- Based on the original [Face-Detection](https://github.com/Shankar2442/Face-Detection) demo by Shankar2442

## License

Add a `LICENSE` file before publishing (MIT is a common choice for small tools like this). Check the licenses of any model files you redistribute.
