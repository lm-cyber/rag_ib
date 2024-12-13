# Dockerfile for the backend app

FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy pyproject.toml and poetry.lock
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root

# Copy the application code
COPY . ./

# Expose the application's port
EXPOSE 8000

# Command to run the app
CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

