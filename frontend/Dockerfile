# Dockerfile for the frontend

FROM python:3.12-slim

# Set the working directory
WORKDIR /frontend

# Install Poetry
RUN pip install poetry

# Copy pyproject.toml and poetry.lock
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root

# Copy the frontend code
COPY . ./

# Expose the frontend's port
EXPOSE 7860

# Command to run the frontend
CMD ["python", "gradio"]

