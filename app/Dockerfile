# Stage 1: install dependencies
FROM python:3.9-slim AS build

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Stage 2: copy source & runtime
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependencies from build
COPY --from=build /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy your app code
COPY . .

# Expose port and launch Uvicorn
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
