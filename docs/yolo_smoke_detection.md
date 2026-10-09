# internship_inML
This repository contains my work during my ML internship.

📂 Folder Overview
/images/ → Contains screenshots and reference visuals.
/notebooks/ → Will include Python notebooks or scripts used during my internship.
README.md → This document explains the structure and purpose of the repository.
🧠 Project Summary
During this internship, I explored the following:

# task 1: Object Detection and Segmentation using YOLOv8

This repository contains my work for the **Machine Learning Internship** project — focusing on **object detection and segmentation** using the **YOLOv8** model from Ultralytics.

---

## 🚀 Project Overview

The goal of this project is to perform **object detection and segmentation** on real-world images using the YOLOv8 model.

### Tasks Completed:

* Installed and configured the YOLOv8 environment
* Ran object detection on sample images (`classroom.jpg`, `office.jpg`)
* Visualized bounding boxes and segmentation masks
* Compared detection vs segmentation results

---

## 🧩 Tools & Libraries

| Tool                   | Purpose                         |
| ---------------------- | ------------------------------- |
| **Python 3.x**         | Programming language            |
| **Ultralytics YOLOv8** | Object detection & segmentation |
| **OpenCV**             | Image processing                |
| **Matplotlib**         | Visualization                   |
| **NumPy**              | Array operations                |

---

## 📁 Repository Structure

```
internship_inML/
│
├── code/
│   ├── detection_classroom.py
│   ├── segmentation_office.py
│
├── images/
│   ├── classroom.jpg
│   ├── office.jpg
│   └── results/
│       ├── classroom_detected.jpg
│       ├── office_segmented.jpg
│
├── requirements.txt
└── README.md
```

---

## 📷 Sample Results

| Image                  
| ---------------------- 
| classroom_detected.jpg 
| office_segmented.jpg   
---

## 🧠 Next Steps

* Experiment with different YOLOv8 models (e.g., `yolov8m`, `yolov8l`)
* Evaluate performance using metrics like Precision, Recall, and mAP
* Try real-time detection using webcam feed

---




## task 2:  Video Detection using YOLOv8

In this section, we take a short video clip and perform object detection frame-by-frame.

### 🧩 Step 1: Extract frames from the video

We use **FFmpeg** to split a video into sequential images:

```bash
ffmpeg -i input.mp4 frames/frame_%04d.jpg
```
### 🎥 Video Detection Results
- [Input video (Google Drive link)]([https://drive.google.com/...](https://drive.google.com/file/d/1hm5oYblcNJ7Kw0asFL1eLytWiEI98Os3/view?usp=drive_link))
- [Processed output video (Google Drive link)](https://drive.google.com/file/d/1GHijquTaxzF1daGmt_42MJWdRRpEcneJ/view?usp=drive_link)

* `input.mp4` → your video file
* `frames/frame_%04d.jpg` → output folder and image pattern

### 🧠 Step 2: Run YOLOv8 object detection on all frames

```bash
yolo detect predict model=yolov8n.pt source="frames/" save=True
```

This generates processed images (with bounding boxes) inside the `runs/detect/predict/` folder.

### 🎞️ Step 3: Combine processed frames back into a video

We use FFmpeg again to stitch them together:

```bash
ffmpeg -framerate 24 -i runs/detect/predict/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p output.mp4
```

* `output.mp4` will be your final detected video
* You can play it using any media player

### 🧾 Output

* **Input video:** `input.mp4`
* **Processed video:** `output_peopleWalking.mp4`

This demonstrates how YOLOv8 can handle not just static images, but dynamic video streams by processing frames in sequence.



## vision task: Outdoor Smoke Detection using YOLOv8

### Problem Statement
Outdoor smoke originating from factories, vehicles, and wildfires contributes significantly to air pollution and poses serious risks to human health and the environment.
The goal of this project is to design a computer vision system that can automatically detect outdoor smoke in videos, with minimal human intervention.

### Project Objective
- Detect smoke in real-world outdoor videos
- Train a custom object detection model using transfer learning
- Evaluate the model on a held-out validation set of annotated frames
- Produce a single consolidated demo video as proof of performance

### Approach
1. Collected real-world outdoor smoke videos (factory emissions, vehicular smoke, wildfire smoke).
2. Extracted diverse frames from videos using FFmpeg.
3. Annotated smoke regions manually using Label Studio.
4. Created a custom YOLO-format dataset (train/validation split).
5. Performed transfer learning from a COCO-pretrained YOLOv8n model (6 training experiments).
6. Added frame-to-frame object tracking to stabilise detections across video frames.
7. Ran detection on the source clips and merged the outputs into a single demonstration video.

### Frame Extraction
Used FFmpeg to extract frames from videos.
Special care was taken to avoid near-duplicate frames (consecutive frames at high FPS look almost identical and cause overfitting).
Selected diverse frames manually to improve generalization.

### Annotation (Major Challenge)
Installed and used Label Studio for manual annotation and labeled smoke regions with bounding boxes.
Main challenges:
- Configuring Label Studio to correctly load local image paths
- Avoiding over-annotation of visually similar frames
- Smoke has no sharp edges, so deciding where a box should end is inherently fuzzy

Final dataset: **151 annotated frames** (120 train / 31 validation) from vehicle-exhaust, factory and wildfire clips.

### Dataset Preparation
Converted annotations into YOLO format and organized the dataset into `images/train`, `images/val`, `labels/train`, `labels/val`, with a custom `data.yaml` for training.

### Training Setup (final run, `train6`)
| Setting | Value |
| --- | --- |
| Base model | YOLOv8n (COCO-pretrained) |
| Epochs | 30 |
| Image size | 640 |
| Batch size | 16 |
| Initial learning rate | 0.01 |
| Augmentation | Mosaic (off for the last 10 epochs), horizontal flip, HSV colour jitter |

### Results
Validation metrics after the final epoch (`train6/results.csv`):

| Metric | Value |
| --- | --- |
| Precision | 0.32 |
| Recall | 0.23 |
| mAP@50 | 0.20 |
| mAP@50-95 | 0.08 |

An earlier run on a smaller dataset (about 70 training images) reached mAP@50 = 0.17, so adding labelled data was the main driver of improvement.

**Limitations (honest assessment):**
- 151 frames is a small dataset for a visually ambiguous class like smoke.
- Training and validation frames come from the same source clips, so validation scores are likely optimistic. A video-level split would give a fairer estimate.
- The demo video is run on the same clips the frames came from, so it shows the pipeline working end to end rather than generalization to new footage.

**Next steps:** more diverse footage, a written annotation guide for consistent boxes, a video-level train/val split, and trying segmentation masks instead of boxes.

### Repository Structure
```
├── README.md
├── final_output_smoke_detection.mp4   # merged demo video
├── input.mp4, merged_input.mp4        # source video inputs
├── train6/
│   ├── args.yaml                      # training configuration of the final run
│   ├── results.csv                    # per-epoch losses and metrics
│   └── weights/
│       ├── best.pt                    # best checkpoint (final smoke model)
│       └── last.pt
└── vision_Task_README.md
```

⚠️ **Note on missing files:** due to GitHub file size limits, the annotated images, labels, raw extracted frames and `data.yaml` are not uploaded. They were used locally for annotation, training and validation, and are reflected in the trained weights and output video.

### Tools & Technologies
- Python
- YOLOv8 (Ultralytics)
- Label Studio
- FFmpeg
- OpenCV
- GitHub

### Key Learnings
- Dataset quality is more important than dataset size, but for a fuzzy class like smoke, more labelled data still helped most.
- Annotation is the most time-consuming and critical step.
- How you split train and validation data decides whether your metrics can be trusted.
- End-to-end pipelines are significantly harder than running pre-trained demos.

### Output
- Final merged video: `final_output_smoke_detection.mp4`

*Created and maintained by [Rhuthvija](https://github.com/Rhuthvija)* ✨
