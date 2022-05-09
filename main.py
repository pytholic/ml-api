# Standard library imports.
import re
import subprocess

# Related third party imports.
from starlette.routing import Route
from fastapi import FastAPI


def get_git_describe():
    try:
        # version tag like 'v1.2.3' should be placed in any of ancesstor commit.
        label = subprocess.check_output(['git', 'describe']) \
            .strip() \
            .decode('utf-8')
        if label.startswith('v'):
            return label
    except subprocess.CalledProcessError:
        return 'NO git version tag'


# preparing API web service
app = FastAPI(
    title='ML APIs of SKIA',
    version=get_git_describe(),
    description='ML prediction APIs of SKIA Co., Ltd.',
)


# imports sub routers
# TODO: automatically include all routers in `api` directory.
if True:  # to avoid rearrange code below by `autopep8` formatter
    from api import Sample
    app.include_router(Sample.router)
    from api import Status
    app.include_router(Status.router)
    from api import landmark
    app.include_router(landmark.router)


# making case-insensitive routing path. ref: https://github.com/tiangolo/fastapi/issues/826#issuecomment-569831634
for route in app.router.routes:
    if isinstance(route, Route):
        route.path_regex = re.compile(route.path_regex.pattern, re.IGNORECASE)


@app.get("/")
def main():
    return {'message': 'Hello world!'}


# run and debug in IDE
#   or run `uvicorn main:app --reload` in terminal
# and open http://localhost:8000/docs to test
if __name__ == "__main__":
    import uvicorn
    file_name_without_ext = 'main'
    uvicorn.run(f'{file_name_without_ext}:app', host='0.0.0.0', port=8000, reload=True)
