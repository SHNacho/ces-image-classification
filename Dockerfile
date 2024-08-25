FROM python:3.12.4-slim-bullseye

# Set environment variables.
ENV PYTHONWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV PYTHONOPTIMIZE 1
ENV ACCEPT_EULA=Y

RUN apt-get -y update

# Set working directory.
WORKDIR /code

# Copy dependencies.
COPY ./requirements.txt /code/

# Install dependencies.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Copy project.
COPY ./app /code/app
COPY ./templates /code/templates

EXPOSE 8000

ENTRYPOINT [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

