# Image for the tuner Job (the agent + benchmark driver).
# The model-under-test runs in a separate, Nebius-managed vLLM Endpoint container.
FROM python:3.11-slim
# The tuner drives endpoint create/teardown through the nebius CLI; inside a
# Serverless Job it authenticates from NEBIUS_IAM_TOKEN (see job-entrypoint.sh).
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash \
    && ln -s /root/.nebius/bin/nebius /usr/local/bin/nebius
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .
COPY job-entrypoint.sh /usr/local/bin/job-entrypoint.sh
RUN chmod +x /usr/local/bin/job-entrypoint.sh
ENTRYPOINT ["job-entrypoint.sh"]
