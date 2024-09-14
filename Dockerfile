# Use the official Python image.
FROM python:3.9-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file to the container.
COPY requirements.txt .

# Install dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container.
COPY . .

# Define the default command to run the app.
CMD ["python", "main.py"]
