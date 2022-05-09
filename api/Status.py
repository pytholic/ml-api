# Standard library imports.
import os
import sys

# Related third party imports.
import psutil
from fastapi import APIRouter, Request

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

# APIs #######################################################################


@router.get("/memory",
            summary='memory status of this service')
def memory():
    process = psutil.Process(os.getpid())
    # see https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_full_info
    mem = process.memory_full_info()
    return {'residentSetSize': mem.rss, 'virtualMemorySize': mem.vms, 'uniqueSetSize': mem.uss}


@router.get("/version",
            summary="service version")
def version(request: Request):
    return request.app.version  # https://github.com/tiangolo/fastapi/issues/702


####################################################################### APIs #

# Test #######################################################################


def test(argv):
    print(memory())
    # TOOD: test `version` function with creating Request object


if __name__ == '__main__':
    test(sys.argv)
####################################################################### Test #
