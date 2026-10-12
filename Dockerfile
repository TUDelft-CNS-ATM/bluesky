FROM python:3.13-slim

# Update system packages to reduce vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# setup user and workdir
RUN addgroup runner && adduser --system --disabled-password --home /app --ingroup runner runner

# Expose output directory as a volume
RUN mkdir -p /data
VOLUME /data

# Install bluesky and its dependencies
RUN pip install --upgrade pip && pip install bluesky-simulator

# Run bluesky
USER runner
WORKDIR /data
ENTRYPOINT [ "bluesky" , "--detached", "--workdir", "/data" ]