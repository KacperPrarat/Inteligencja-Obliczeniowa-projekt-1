import json
import json

def convert_to_yolov8_format(annotations, images):
    yolov8_annotations = []
    for annotation in annotations:
        try:
            bbox = annotation['bbox']
            keypoints = annotation['keypoints']

            # Find corresponding image dimensions
            image_info = next((image for image in images if image['id'] == annotation['image_id']), None)
            if image_info is None:
                print(f"No corresponding image found for annotation {annotation['id']}. Skipping...")
                continue

            image_width = image_info['width']
            image_height = image_info['height']

            # Convert bounding box to YOLO format
            x_center = (bbox[0] + bbox[2] / 2) / image_width
            y_center = (bbox[1] + bbox[3] / 2) / image_height
            width = bbox[2] / image_width
            height = bbox[3] / image_height

            # Convert key points to YOLO format (if needed)
            # Normalize key point coordinates

            # Construct YOLO annotation string
            yolo_annotation = f"{annotation['category_id']} {x_center} {y_center} {width} {height} {' '.join(map(str, keypoints))}"
            yolov8_annotations.append(yolo_annotation)
        except Exception as e:
            print(f"Error processing annotation: {e}")

    return yolov8_annotations

# def main():
#     # Define input and output file paths
#     json_file_path = 'annotations.json'
#     output_file_path = 'annotations_yolo.txt'

#     # Load JSON file
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)

#     # Extract annotations and images
#     annotations = data['annotations']
#     images = data['images']

#     # Convert annotations to YOLOv8 format
#     yolov8_annotations = convert_to_yolov8_format(annotations, images)

#     # Save YOLOv8 annotations to text file
#     with open(output_file_path, 'w') as f:
#         f.write('\n'.join(yolov8_annotations))

#     print(f"Annotations converted to YOLOv8 format and saved to {output_file_path}")

# if __name__ == "__main__":
#     main()


def main():
    # Define input and output file paths
    json_file_path = 'CCTVAnnotations.json'
    output_file_path = 'annotations_yolo.txt'

    # Load JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract annotations and images
    annotations = data['annotations']
    images = data['images']

    # Convert annotations to YOLOv8 format
    yolov8_annotations = convert_to_yolov8_format(annotations, images)

    # Save YOLOv8 annotations to text file
    with open(output_file_path, 'w') as f:
        f.write('\n'.join(yolov8_annotations))

    print(f"Annotations converted to YOLOv8 format and saved to {output_file_path}")

if __name__ == "__main__":
    main()
