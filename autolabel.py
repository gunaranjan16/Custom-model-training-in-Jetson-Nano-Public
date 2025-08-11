import os
import cv2
from ultralytics import YOLO
import xml.etree.ElementTree as ET

# Load YOLO model
model = YOLO(".pt")  # Update the model file if needed

def filter_overlapping_boxes(detections, iou_threshold=0.5):
    """Filter overlapping bounding boxes using IoU."""
    filtered_detections = []
    for i, box1 in enumerate(detections):
        keep = True
        for j, box2 in enumerate(detections):
            if i != j:
                iou = calculate_iou(box1, box2)
                if iou > iou_threshold and box1['confidence'] < box2['confidence']:
                    keep = False
                    break
        if keep:
            filtered_detections.append(box1)
    return filtered_detections

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1 = max(box1['xmin'], box2['xmin'])
    y1 = max(box1['ymin'], box2['ymin'])
    x2 = min(box1['xmax'], box2['xmax'])
    y2 = min(box1['ymax'], box2['ymax'])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    area_box2 = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
    union = area_box1 + area_box2 - intersection

    return intersection / union if union > 0 else 0

def create_pascal_voc_xml(output_folder, image_filename, image_shape, detections):
    """Create a Pascal VOC XML file for the given image and detections."""
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = os.path.basename(output_folder)
    ET.SubElement(root, "filename").text = image_filename

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image_shape[1])
    ET.SubElement(size, "height").text = str(image_shape[0])
    ET.SubElement(size, "depth").text = str(image_shape[2])

    for detection in detections:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = detection['class']
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(detection['xmin'])
        ET.SubElement(bbox, "ymin").text = str(detection['ymin'])
        ET.SubElement(bbox, "xmax").text = str(detection['xmax'])
        ET.SubElement(bbox, "ymax").text = str(detection['ymax'])

    tree = ET.ElementTree(root)
    xml_filename = os.path.join(output_folder, os.path.splitext(image_filename)[0] + ".xml")
    tree.write(xml_filename)

def annotate_images(input_folder, output_folder, labels_file, max_objects_file, specific_class=None):
    """Annotate images using YOLO and generate Pascal VOC XML files."""
    os.makedirs(output_folder, exist_ok=True)

    labels = set()
    max_objects_per_frame = {}

    for image_filename in os.listdir(input_folder):
        if not image_filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(input_folder, image_filename)
        image = cv2.imread(image_path)
        image_shape = image.shape

        results = model(image)

        detections = []
        for box in results[0].boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0]  # Coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = box.cls[0]  # Class ID
            class_name = model.names[int(class_id)]  # Class name

            detections.append({
                'class': class_name,
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax),
                'confidence': float(confidence)
            })

        if specific_class:
            detections = [d for d in detections if d['class'] == specific_class]

        for detection in detections:
            labels.add(detection['class'])

        filtered_detections = filter_overlapping_boxes(detections)

        for detection in filtered_detections:
            obj_class = detection['class']
            max_objects_per_frame[obj_class] = max(
                max_objects_per_frame.get(obj_class, 0),
                sum(1 for d in filtered_detections if d['class'] == obj_class)
            )

        create_pascal_voc_xml(output_folder, image_filename, image_shape, filtered_detections)

    with open(labels_file, "w") as f:
        f.writelines(f"{label}\n" for label in sorted(labels))

    with open(max_objects_file, "w") as f:
        for cls, count in max_objects_per_frame.items():
            f.write(f"{cls}: {count}\n")

if _name_ == "_main_":
    input_folder = "C:\\Users\\ASHOK\\python\\jetson-train\\data\\auto\\images"  # Update with your input folder path
    output_folder = "C:\\Users\\ASHOK\\python\\jetson-train\\data\\auto\\annotations"  # Update with your output folder path
    labels_file = "C:\\Users\\ASHOK\\python\\jetson-train\\data\\auto\\labels.txt"
    max_objects_file = "C:\\Users\\ASHOK\\python\\jetson-train\\data\\auto\\max_obj.txt"

    annotate_images(input_folder, output_folder, labels_file, max_objects_file)
