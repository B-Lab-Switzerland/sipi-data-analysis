FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
	libglib2.0-0\
    libdbus-1-3\
    libatk1.0-0\
    libatk-bridge2.0-0\
    libcups2\
    libxkbcommon0\
    libatspi2.0-0\
    libxcomposite1\
    libxdamage1\
    libxfixes3\
    libxrandr2\
    libgbm1\
    libpango-1.0-0\
    libasound2\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /analysis_app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install playwright
RUN playwright install
RUN playwright install-deps

# Copy notebooks and data
COPY . .

# Expose port for Jupyter
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

