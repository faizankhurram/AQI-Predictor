"""Hopsworks login helper — correct SaaS host for Serverless / cloud API."""

import os
import socket

# SDK 3.7 defaults to c.app.hopsworks.ai, which has no public DNS record.
DEFAULT_HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"


def get_hopsworks_host() -> str:
    return os.environ.get("HOPSWORKS_HOST", DEFAULT_HOPSWORKS_HOST)


def login_hopsworks(project: str | None = None, api_key_value: str | None = None):
    """Login to Hopsworks using env credentials and a resolvable SaaS host."""
    import hopsworks

    host = get_hopsworks_host()
    project = project or os.environ["HOPSWORKS_PROJECT"]
    api_key_value = api_key_value or os.environ["HOPSWORKS_API_KEY"]

    try:
        socket.getaddrinfo(host, 443)
    except OSError:
        raise ConnectionError(
            f"Cannot resolve Hopsworks host '{host}'. "
            f"Set HOPSWORKS_HOST={DEFAULT_HOPSWORKS_HOST} in .env "
            f"(c.app.hopsworks.ai is not in public DNS)."
        ) from None

    return hopsworks.login(project=project, api_key_value=api_key_value, host=host)
