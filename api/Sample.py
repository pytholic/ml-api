# Standard library imports.
import base64
import os
import sys

# Related third party imports.
import onnxruntime
import numpy as np
import cv2
from fastapi import APIRouter, Response, UploadFile
from pydantic import BaseModel
from typing import Optional
from functools import lru_cache  # pip install cachetools

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


# logic ######################################################################
#
#   try to represent return type(type hints https://docs.python.org/3/library/typing.html)
#   and more description in docstring comment. definitely it helps a lot to teammates.
#   in other hands, API function should describe details by @router decorator
#   to expose them on web interface.
#
#   try to keep independent(less-coupled) unit functions, which makes easier test.
#

# using cache for deterministic heavy functions with @lru_cache decorator
@lru_cache(maxsize=2)  # cache recent 2 volumes
def __load_Int16Volume_cache(file_path: str) -> Int16Volume:
    """
    trying to use Int16Volume(.vol) from cached memory
    or loads from given file path if not exists.
    """
    return Int16Volume.load(file_path)


@lru_cache(maxsize=2)
def __inference_session_chache(file_path: str) -> onnxruntime.InferenceSession:
    """load onnx session and cache."""
    return onnxruntime.InferenceSession(file_path, None)


def __get_slice_as_png_bytes(volume_path_from_resource: str, slice_index: int) -> bytes:
    """
    extract the single .png image content from the Int16Volume file.

    Returns:
        bytes of the image content
        None if the image encoding fails
    """
    vol_path = os.path.join(RESOURCE_DIR, volume_path_from_resource)
    vol = __load_Int16Volume_cache(vol_path)
    slice = vol.getSlice(slice_index)

    # opencv deson't support 16bit grayscale PNG image writing.
    #   8bit grayscale conversion instead.
    normalized_slice = cv2.normalize(
        slice, None, 0, 255, cv2.NORM_MINMAX,
        dtype=cv2.CV_8UC1)

    success, img = cv2.imencode('.png', normalized_slice)
    if success == False:
        return None
    return img.tobytes()


def __predict_mnist(file_content: bytes) -> int:
    """
    classify a number from hand-wrting image content.

    Returns:
        the prediction result that's in range 0 ~ 9
    """

    # NO IDEA the reason of this code
    file_content = np.frombuffer(file_content, dtype=np.uint8)

    # preprocessing
    img = cv2.imdecode(file_content, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_AREA)
    average = img.mean(axis=0).mean(axis=0)
    if average > 128:  # assuming white background
        img = (255 - img)  # need to be inverted(black background)
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32FC1)
    input = np.reshape(img, (1, 1, 28, 28))

    # preparing pretrained model
    model_file_path = os.path.join(RESOURCE_DIR, 'ml-model/mnist-8.onnx')
    session = __inference_session_chache(model_file_path)
    output_name = session.get_outputs()[0].name
    input_name = session.get_inputs()[0].name

    # predection
    result = session.run([output_name], {
                         input_name: input})
    prediction_result = int(np.argmax(np.array(result).squeeze(), axis=0))
    return prediction_result


###################################################################### logic #

# APIs #######################################################################
#
#   functions for end-users(remote accessors).
#   implementation should be simple by calling internal logic fuctions
#   to make easier test without running browser.
#

@router.get("/",
            summary=f'showing description of this({file_name_without_ext}) API set')
def main():
    return {'message': 'Hello, Sample!'}


@router.get("/dicom/{sliceIndex}",
            summary=f'return base64 encoded .png image for the given slice index')
def get_dicom_slice(slice_index: int):
    base64_encoded = base64.b64encode(
        __get_slice_as_png_bytes('volume/head-dicom.vol', slice_index))
    return base64_encoded


@router.get("/test/dicom/{sliceIndex}",
            responses={  # ref: https://fastapi.tiangolo.com/advanced/additional-responses/#additional-media-types-for-the-main-response
                200: {'content': {'image/png': {}}}
            },
            # to prevent "application/json" response. ref: https://github.com/tiangolo/fastapi/issues/3258
            response_class=Response,
            summary='get a slice image from sample DICOM volume')
def get_dicom_slice_test(slice_index: int):
    """
        outputs browser friendly(human readable) result
    """
    content = get_dicom_slice(slice_index)
    return Response(content=base64.b64decode(content), media_type='image/png')


class __SkinMaskRequestBody(BaseModel):
    slice_index: int
    comment: Optional[str] = None


# RequestBody(body form) cannot be used with GET/HEAD verb
@router.post("/mask",
             summary=f'return json which contains base64 encoded .png image of skin mask for the given slice index')
def get_mask_slice(req: __SkinMaskRequestBody):
    base64_encoded = base64.b64encode(__get_slice_as_png_bytes(
        'volume/head-skin.vol', req.slice_index))
    return {'sliceIndex': req.slice_index, 'comment': req.comment, 'content': base64_encoded}


@router.post("/test/mask",
             responses={  # ref: https://fastapi.tiangolo.com/advanced/additional-responses/#additional-media-types-for-the-main-response
                 200: {'content': {'image/png': {}}}
             },
             # to prevent "application/json" response. ref: https://github.com/tiangolo/fastapi/issues/3258
             response_class=Response,
             summary='get a slice image from sample DICOM volume')
def get_mask_slice_test(req: __SkinMaskRequestBody):
    result = get_mask_slice(req)
    return Response(content=base64.b64decode(result['content']), media_type='image/png')


class __MnistRequestBody(BaseModel):
    base64_encoded_image_file: str


@router.post("/mnist",
             summary=f'return mnist prediction(0 to 9) value from the base64 encoded image')
async def mnist(req: __MnistRequestBody):
    image = base64.b64decode(req.base64_encoded_image_file)
    prediction_result = __predict_mnist(image)
    return {'prediction': prediction_result}


@router.post("/test/mnist",
             summary=f'return mnist prediction(0 to 9) value from the input handwriting number image')
# pip install python-multipart required to use Form(file-upload) data
async def mnist_test(file: UploadFile):
    content = await file.read()
    base64_content = base64.b64encode(content)
    req = __MnistRequestBody(base64_encoded_image_file=base64_content)
    return await mnist(req)


####################################################################### APIs #


# Test #######################################################################
#
#   you might want to test logic functions and APIs without running web-service.
#   you can run just this file directly with your test function like below.
#

def test(argv):
    test_num = 9
    file_path = os.path.join(RESOURCE_DIR, f'test-input/mnist/{test_num}.png')
    content = open(file_path, 'rb').read()
    result = __predict_mnist(content)
    print(f'predicted number is {result}')


if __name__ == '__main__':
    test(sys.argv)
####################################################################### Test #
