# Image for the tuner Job (the agent + benchmark driver).
# The model-under-test runs in a separate, Nebius-managed vLLM Endpoint container.
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["tuner"]
