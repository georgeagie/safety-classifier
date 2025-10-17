# Use a slim Python 3.11 image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install necessary system dependencies for Matplotlib (Qt5Agg)
# Note: For many scientific applications, an older base image like 'ubuntu:20.04' might be necessary 
# to install all the required libraries for Qt, but 'slim' works well for core Python. 
# We install the common library needed for many plot outputs.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# We install 'pyqt5' as the Qt5Agg backend requires it.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
# Assuming the user names their simulation file 'simulation.py'
# NOTE: Ensure you rename your code file to 'simulation.py' locally.
COPY simulation.py .

# Command to run the script (can be overridden with 'docker run ...')
# We use the Agg backend by default for robustness in containers. 
# You need to modify simulation.py to ensure it saves the plot to a file 
# instead of relying on interactive 'plt.show()'.
CMD ["python", "simulation.py"]
