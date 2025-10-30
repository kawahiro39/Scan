# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Allow statements and log messages to be sent straight to the logs
# without being buffered first.
ENV PYTHONUNBUFFERED True

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt .

# Install the dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container.
COPY . .

# Expose the port that the application will listen on.
EXPOSE 8080

# Run the application.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
