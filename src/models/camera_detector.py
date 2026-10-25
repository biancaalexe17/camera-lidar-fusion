
from ultralytics import YOLO

def load_camera_detector(weights="yolov8n.pt"):
    """Load YOLO once; Ultralytics downloads the weights if needed."""
    return YOLO(weights)

def predict_camera(model, image, confidence=0.25):
    """Return camera boxes, scores, class IDs, and class names."""
    result = model.predict(
        source=image,
        conf=confidence,
        verbose=False,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return {
            "boxes": [],
            "scores": [],
            "labels": [],
            "names": result.names,
        }

    return {
        "boxes": result.boxes.xyxy.cpu().numpy(),
        "scores": result.boxes.conf.cpu().numpy(),
        "labels": result.boxes.cls.cpu().numpy().astype(int),
        "names": result.names,
    }
