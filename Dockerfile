# Use official Python base image
FROM python:3.11

# Working directory in container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose default ports
EXPOSE 8501
EXPOSE 5000

# Run Streamlit app
CMD ["streamlit", "run", "app.py"]
