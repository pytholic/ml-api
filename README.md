# ML-API

ML prediction API service for SKIA.


## Requirements

 * python 3.9+ and packages ; see [Requirements.txt](./Requirements.txt)
    - in case of conda
```
conda create --name ml-api python=3.9
conda activate ml-api
pip install --no-cache-dir --upgrade -r ./requirements.txt
```


## Getting started

### To run on development environment

 1. Run and Debug `main.py` in Visual studio code OR run [`uvicorn main:app --reload`](https://fastapi.tiangolo.com/tutorial/first-steps/) in terminal.
 2. Open `http://localhost:8000/docs` in browser and check APIs, OR `http://localhost:8000/redoc` for [Redoc OpenAPI documentation](https://github.com/Redocly/redoc).
  
### To add a new API set

Keep the source structure simple and separated.

 1. Take a look [`api/Sample.py`](./api/Sample.py) file first.
 2. Add a new python file as the set name.
 3. Copy the sample file.
 4. Replace logic and APIs code bock to your own.
 5. Test your API functions with `test(argv)` function while developing.
 7. Test your API with `main.py` and browser
     1. Add import and router in `# imports sub routers` section
     2. Run a browser and Test APIs of `http://localhost:8000/docs`

### To deploy ml-api service

 1. Clone or checkout this repository
 2. Execute `docker_run.sh`
 

## Background and history

It's very hard and time consuming tasks to implement ML models into end user device or cross-platform service. SKIA has currently 3 different platforms of them. [**iOS app**](https://bitbucket.org/alkee_skia/mars3/), [**Unity based Windows app**](https://bitbucket.org/alkee_skia/mars-processor) and **.NET core service**(cross-platform required). If the prediction is not required to be done in real-time level response like camera input and overlay on it, We could use remote API(network request and response) which serves ML prediction by direct python code.

I dicided to use [FastAPI](https://fastapi.tiangolo.com/) as a web service framework for lots of things to do easy implementation without big efforts.

 * FastAPI
     - [Complete Guide on Rest API with Python and Flask](https://www.analyticsvidhya.com/blog/2022/01/rest-api-with-python-and-flask/)
     - [Deploying ML Models as API using FastAPI](https://www.geeksforgeeks.org/deploying-ml-models-as-api-using-fastapi/)
     - [FLASK에서 FASTAPI로 간 이유](https://tech.madup.com/FastAPI/)
