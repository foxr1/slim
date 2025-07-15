# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir makes the image smaller and the build faster
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Copy the fine-tuned models and source texts into the container
COPY models_quantised/ ./models_quantised/
COPY source_texts/ ./source_texts/

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define the command to run the app using Gunicorn, a production-grade server.
# This is the command that Cloud Run will execute to start the service.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app"]
