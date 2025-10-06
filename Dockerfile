# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create output folder so plots can be saved
RUN mkdir -p /app/output

# Run the simulation
CMD ["python", "main.py"]
