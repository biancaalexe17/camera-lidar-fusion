# Camera–LiDAR Fusion for 3D Object Detection

A research project exploring camera–LiDAR fusion for 3D object detection using the Waymo Open Dataset, a PointPillars LiDAR detector, and a YOLOv8 camera detector.

## Overview

This project investigates how 2D camera detections can be combined with 3D LiDAR detections to support object detection in autonomous driving scenarios.

The pipeline uses LiDAR point clouds to obtain 3D object detections and camera images to obtain 2D detections. LiDAR bounding boxes are geometrically projected into the camera image using sensor calibration. The projected boxes can then be compared with camera detections to establish cross-modal associations.

The project focuses on the geometric projection, detection association, and evaluation stages of the fusion pipeline, including the analysis of projection failures and the effect of camera support on LiDAR detections.

## Objectives

- Process synchronized camera and LiDAR data from the Waymo Open Dataset.
- Run a PointPillars detector on LiDAR point clouds.
- Run a YOLOv8 detector on camera images.
- Project 3D LiDAR bounding boxes into the camera image using calibration data.
- Associate projected LiDAR detections with camera detections.
- Evaluate detection and association behavior using quantitative metrics and visual inspection.
- Validate the pipeline on held-out Waymo contexts.

## Pipeline

The main processing stages are:

1. **Data preparation** — load and convert Waymo camera, LiDAR, calibration, and annotation data.
2. **LiDAR detection** — obtain 3D detections from PointPillars.
3. **Camera detection** — obtain 2D detections from YOLOv8.
4. **Geometric projection** — transform 3D bounding-box corners into the camera coordinate system and project them into image coordinates.
5. **Cross-modal association** — compare projected LiDAR boxes with camera detections using geometric overlap.
6. **Evaluation** — analyze detection counts, projection validity, associations, IoU statistics, and performance against ground-truth annotations.
7. **Visualization** — inspect camera overlays and bird's-eye-view results to understand successes and failure cases.

## Experiments

The repository contains three experiment notebooks.

### Experiment 1 — Custom-Trained PointPillars

Evaluates the fusion pipeline using the custom-trained PointPillars detector. The notebook includes diagnostics for calibration, coordinate transformations, 3D bounding-box projection, and camera–LiDAR association.

### Experiment 2 — Pretrained PointPillars

Evaluates the pipeline using a pretrained PointPillars detector. The experiment investigates how the pretrained detector's outputs behave after projection into the camera image and how they associate with YOLOv8 detections.

### Experiment 3 — Validation on Unseen Waymo Contexts

Evaluates the pipeline on five held-out Waymo contexts. The validation workflow includes data-integrity checks, prediction cleaning, frame-alignment checks, quantitative evaluation against ground truth, comparison of baseline and camera-supported predictions, and qualitative visualization.

Validation contexts are kept separate from model and fusion-parameter tuning.

## Repository Structure

```text
Project2/
├── README.md
├── .gitignore
├── setup.ipynb
├── configs/
├── experiments/
│   ├── experiment_1.ipynb
│   ├── experiment_2.ipynb
│   ├── validation.ipynb
│   ├── run_experiment_1.py
│   ├── run_experiment_2.py
│   └── plot_experiment_1.py
├── src/
│   ├── config.py
│   ├── setup_runtime.py
│   ├── data/
│   ├── evaluation/
│   ├── fusion/
│   └── models/
└── tests/
```

### Main modules

- `src/data/` — data loading, conversion, and validation-data handling.
- `src/models/` — camera and LiDAR detector interfaces.
- `src/fusion/` — projection, matching, association, evaluation, and diagnostic utilities.
- `src/evaluation/` — BEV metrics and validation scripts.
- `experiments/` — experiment notebooks, execution scripts, and plotting utilities.
- `tests/` — tests for configuration and data-loading functionality.
- `configs/` — project configuration files.

## Evaluation

The project uses complementary quantitative and qualitative checks, including:

- **Detection counts** — the number of detections produced by the pipeline.
- **Projection statistics** — the number and proportion of LiDAR boxes that can be projected into the camera image.
- **Association statistics** — matched and unmatched detections and the distribution of overlap scores.
- **BEV IoU** — comparison of 3D object detections with ground-truth boxes in bird's-eye view.
- **Baseline comparison** — comparison of baseline LiDAR predictions with camera-supported predictions.
- **Visual inspection** — camera overlays and bird's-eye-view plots to investigate projection and association failures.

The exact evaluation settings and thresholds are defined by the implementation and experiment configuration.

## Requirements

The project uses Python and relies on libraries and tools for numerical computing, data processing, computer vision, deep learning, and 3D perception.

Depending on the selected experiment, the environment may require:

- Python
- NumPy and pandas
- OpenCV
- PyTorch
- Ultralytics YOLO
- The PointPillars implementation and its dependencies
- Waymo Open Dataset data-processing dependencies
- Additional libraries specified by the project configuration or runtime setup

GPU availability and compatible library versions may affect execution time and model compatibility.

## Setup

1. Clone the repository:

   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd Project2
   ```

2. Create and activate a Python environment appropriate for the project's dependencies.

3. Install the required packages and configure the PointPillars and camera detector dependencies.

4. Configure the paths to the Waymo dataset, model weights, and experiment outputs.

5. Follow the relevant experiment notebook in `experiments/`.

The exact installation commands depend on the PointPillars implementation, Waymo conversion dependencies, and versions used in the working environment. These should be documented once the project's runtime configuration has been consolidated.

## Data and Model Weights

The Waymo Open Dataset, trained model weights, and generated experiment outputs are not included in this repository by default.

Users must obtain the dataset and model weights separately and configure their local paths before running the experiments. Dataset access and use must comply with the applicable dataset terms.

Large model checkpoints, raw sensor data, credentials, and generated outputs should not be committed to Git.

## Limitations

- Camera–LiDAR association depends on calibration accuracy and correct synchronization between sensor measurements.
- Geometric projection can fail when a 3D bounding box lies outside the camera field of view or crosses the camera plane.
- Camera and LiDAR detectors may produce different classes, bounding-box sizes, and confidence scores.
- Association performance depends on the chosen geometric criteria and evaluation settings.
- Results on held-out contexts may differ from those observed during development.

## Reproducibility

The experiment notebooks document the diagnostic process and evaluation workflow. For reproducible runs, use the intended configuration, keep model weights fixed during validation, and preserve the dataset split and evaluation settings.

Record relevant dependency versions, model checkpoints, configuration values, and output locations when reproducing the results.

## Status

This repository contains the implementation and experimental workflows for a camera–LiDAR fusion project. The notebooks cover custom-trained and pretrained LiDAR detection, geometric projection diagnostics, cross-modal association, and validation on unseen Waymo contexts.

Further documentation of the exact installation procedure, model checkpoints, and final evaluation results can be added as the implementation is finalized.
