# Dockerfile for the frontend

FROM python:3.12-slim

# Set the working directory
WORKDIR /frontend

# Copy pyproject.toml and poetry.lock
COPY requirements.txt ./

# Install dependencies
RUN pip install -r requirements.txt

# Copy the frontend code
COPY . ./

# Expose the frontend's port
EXPOSE 7860

# Command to run the frontend
CMD ["python", "gradio.py"]

