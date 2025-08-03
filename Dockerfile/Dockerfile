FROM python:3.10-slim

WORKDIR /app

# Use only the fast requirements file and rename it before installing
COPY requirements-fast.txt ./requirements.txt

# Install only fast dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy rest of the application
COPY . .

EXPOSE 5001

CMD ["python", "app.py"]
