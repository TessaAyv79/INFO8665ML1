# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .
COPY StockSeeker2xx_v1_0812.py .
COPY Fake-Apache-Log-Generator /app/Fake-Apache-Log-Generator

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# # Make port 8501 available to the world outside this container
EXPOSE 8501

# Streamlit uygulamasını başlat
CMD ["streamlit", "run", "StockSeeker2xx_v1_0812.py", "--server.port=8501", "--server.address=0.0.0.0"]

#docker build -t stockseeker-app .
#docker run -p 8502:8502 stockseeker-app
# Docker image oluştur
#docker build -t my-flask-app .
# Docker container başlat
#docker run -p 8502:8502 flask_api