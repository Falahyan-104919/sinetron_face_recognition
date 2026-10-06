import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import time

from flask import request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "apm.log")


def get_apm_logger():
    """Configures and returns a dedicated logger for APM metrics."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger("sinetron_apm")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def setup_apm_logging(app):
    """Sets up APM request tracking and logging on the Flask app."""
    apm_logger = get_apm_logger()

    @app.before_request
    def apm_before_request():
        request._start_time = time.perf_counter()

    @app.after_request
    def apm_after_request(response):
        try:
            start_time = getattr(request, "_start_time", None)
            if start_time is None:
                start_time = time.perf_counter()
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            level = (
                "error"
                if response.status_code >= 500
                else ("warn" if response.status_code >= 400 else "info")
            )

            payload = {
                "level": level,
                "log_type": "apm_http_metric",
                "service_name": "sinetron-face-recognition",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "message": f"HTTP {request.method} {request.path} {response.status_code} in {duration_ms}ms",
                "http": {
                    "method": request.method,
                    "route": request.endpoint or request.path,
                    "url": request.url,
                    "status_code": response.status_code,
                    "response_time_ms": duration_ms,
                    "is_slow": duration_ms > 1000.0,
                },
                "performance": {
                    "duration_ms": duration_ms,
                },
                "client": {
                    "ip": request.headers.get("X-Forwarded-For", request.remote_addr),
                    "user_agent": request.headers.get("User-Agent", "Unknown"),
                },
            }

            apm_logger.info(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            app.logger.error(f"Failed to write APM log: {e}")

        return response
