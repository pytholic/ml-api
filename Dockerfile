# ref: https://bitbucket.org/alkee_skia/ml-api/issues/2

# the official Python base image.
FROM python:3.9

# environments
ENV APP_HOME /app
EXPOSE 80

# python packages
COPY ./ $APP_HOME/
RUN pip install --no-cache-dir --upgrade -r $APP_HOME/requirements.txt

# extra native library dependencies of the python packages
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# run the ml-api service
WORKDIR $APP_HOME
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
