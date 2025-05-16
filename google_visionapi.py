import os
import sys
from google.cloud import vision_v1 as vision
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"object_detection.json"

def localize_objects(path):
    client = vision.ImageAnnotatorClient()

    # Read image content
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Call the object localization method
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {:.2f})'.format(object_.name, object_.score))  # 🔧 Fix: .score not .store


if len(sys.argv) < 2:
    print("Usage: python script.py path_to_image.jpg")
    sys.exit(1)

path = sys.argv[1]
localize_objects(path)

