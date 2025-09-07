# Use official Python runtime with slim variant for smaller image size
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=TRUE

# Expose the port (optional but good practice)
EXPOSE 8080

# Run as non-root user for security
RUN useradd -m -r appuser && chown -R appuser /app
USER appuser

# Command to run the application
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]