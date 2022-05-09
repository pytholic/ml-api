import os
import sys
import base64

import mediapipe
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi import File, UploadFile, Response
from pydantic import BaseModel
from functools import lru_cache

# common script for API router ###############################################

# to import local sourcecode level packages
if __name__ == '__main__':
    # for test environment(running with this single .py file alone)
    parent_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)

# import source level package(Local application/library specific imports)
from skia.Int16Volume import Int16Volume

# using this file name as a router name(first url path)
file_name_without_ext = os.path.basename(__file__).split('.')[0]
router = APIRouter(
    prefix=f"/{file_name_without_ext}",
    tags=[file_name_without_ext],
)

# use this resource path with `os.path.join` function
#   for any testing or service environment (main.py or this python file only testing)
RESOURCE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../resource'))

############################################### common script for API router #

### Utility functions ###


def adjust_gamma(image: np.uint8, gamma=1.0):

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def read_image(image_encoded):
    img = np.frombuffer(image_encoded, dtype=np.uint8)
    cv2_img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return cv2_img


# Function to draw sample keypoints
def draw_keypoints(keypoints: dict, image: np.uint8):
    for landmark in keypoints.items():
        cv2.circle(image, (int(landmark[1][0]), int(landmark[1][1])),
                   radius=3, color=(0, 0, 255), thickness=-1)
        cv2.putText(image, str(landmark[0]), org=(int(landmark[1][0]), int(landmark[1][1])),
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(255, 0, 0))
    return image

# Test function to visualize all landmarks


def visualize_landmarks(landmarks: list, image: np.uint8):
    for landmark in landmarks:
        cv2.circle(image, (int(landmark[0]), int(landmark[1])),
                   radius=2, color=(0, 0, 255), thickness=-1)
    return image

### Processing function ###


def preprocess(image: np.uint8, gamma=0.4):
    smoothing = 25  # Range between 0 and 200
    edge_preserve = 0.15  # Range between 0 and 1
    orig_size = (image.shape[1], image.shape[0])
    input_size = (224, 224)  # Size of input image for the model
    resized = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
    adjusted = adjust_gamma(resized, gamma=gamma)
    enhanced = cv2.detailEnhance(
        adjusted, sigma_s=smoothing, sigma_r=edge_preserve)
    return enhanced, orig_size

### Predict function ###


def predict_facial_landmark(image: np.uint8, orig_size: tuple):
    mp_face_mesh = mediapipe.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, refine_landmarks=True)
    results = face_mesh.process(image)

    # Handle scenario when no landmarks are found
    if results.multi_face_landmarks is None:
        return None

    landmarks = results.multi_face_landmarks[0]
    lmarks = []
    for landmark in landmarks.landmark:
        x = landmark.x
        y = landmark.y
        # Denormalize the landmarks
        denormalized_x = int(x * orig_size[0])
        denormalized_y = int(y * orig_size[1])
        lmarks.append((denormalized_x, denormalized_y))
    return lmarks


class LandmarkRequestBody(BaseModel):
    base64_encoded_image_file: str


@router.post("/face")
def facial_landmark(req: LandmarkRequestBody):
    img = base64.b64decode(req.base64_encoded_image_file)
    cv2_img = read_image(img)
    image_processed, orig_size = preprocess(cv2_img)
    prediction = predict_facial_landmark(image_processed, orig_size)
    return {"prediction": prediction}


@router.post("/test/face")
async def facial_landmark_test(file: UploadFile = File(..., description="Select an input image.")):
    """
    This function will load an image and predict landmarks.
    """

    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        raise HTTPException(
            status_code=500, detail="Image should be JPEG or PNG")

    # Read the file uploaded by the user
    _bytes = await file.read()

    # Predict landmarks
    base64_content = base64.b64encode(_bytes)
    req = LandmarkRequestBody(base64_encoded_image_file=base64_content)
    prediction = facial_landmark(req)
    for _, value in prediction.items():
        if value is None:
            raise HTTPException(
                status_code=500, detail="Facial landmarks could not be found.")
        else:
            return prediction


@lru_cache(maxsize=1)
@router.post("/reference/face", responses={200: {'content': {'image/png': {}}}}, response_class=Response)
def key_points():

    keypoints = {}
    key_idx = [
        1,              # nose tip
        205,            # right cheek
        425,            # left cheek
        33, 133,        # right eye`
        263, 362,       # left eye
        168,            # midway between eyes
        78, 308,        # lip edges
        11,             # upper lip
        14,             # lower lip
        70, 55,         # right eyebrow
        300, 285        # left eyebrow
    ]

    sample = cv2.imread(
        os.path.join(RESOURCE_DIR, 'test-input/head-heightmap/1.png'),
        cv2.IMREAD_COLOR)
    image, orig_size = preprocess(sample)
    prediction = predict_facial_landmark(image, orig_size)

    # Extract key landmarks from predicted landmarks
    for idx, landmark in enumerate(prediction):
        if idx in key_idx:
            keypoints[idx] = landmark

    resized = cv2.resize(image, orig_size)
    result = draw_keypoints(keypoints, resized)
    result = cv2.resize(result, (512, 512))
    result = cv2.imencode('.png', result)[1].tobytes()

    return Response(content=result, media_type='image/png')


# Test #######################################################################

def test(argv):
    test_num = 2
    test_file_path = os.path.join(
        RESOURCE_DIR, f'test-input/head-heightmap/{test_num}.png')
    test_file_content = open(test_file_path, 'rb').read()
    req = LandmarkRequestBody(
        base64_encoded_image_file=base64.b64encode(test_file_content))
    prediction = facial_landmark(req)
    prediction_list = list(prediction.values())
    flat = [subitem for item in prediction_list for subitem in item]
    sample = cv2.imread(test_file_path)
    result = visualize_landmarks(flat, sample)
    cv2.imshow('Result', result)
    cv2.waitKey()

    # content = np.frombuffer(key_points().body, dtype=np.uint8)
    # img = cv2.imdecode(content, cv2.IMREAD_COLOR)
    # cv2.imshow('reference ', img)
    # cv2.waitKey()


if __name__ == '__main__':
    test(sys.argv)
####################################################################### Test #
