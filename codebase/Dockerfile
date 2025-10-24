FROM python:3.11-slim

# environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1

# working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


COPY . /app

# Streamlit default port
EXPOSE 8501

# Default environment variables
# Mode can be: "gemini" or "local"
ENV MODEL_MODE=gemini
ENV GEMINI_API_KEY="your-api-key-here"
ENV MODEL_NAME="gemini-1.5-pro"

# run streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
