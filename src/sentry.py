"""Optional Sentry integration for error tracking and monitoring.

This module provides optional Sentry SDK integration for the Webex AI Bot.
If sentry-sdk is not installed or SENTRY_DSN is not configured, the bot
will continue to work normally without any monitoring.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    """Parse a boolean value from an environment variable string.

    Args:
        value: The string value to parse (e.g., "true", "false", "1", "0")
        default: Default value if the string is None or empty

    Returns:
        The parsed boolean value
    """
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("true", "1", "yes", "on")


def init_sentry() -> bool:
    """Initialize Sentry SDK if available and configured.

    Environment Variables:
        SENTRY_DSN: Required. The Sentry DSN for your project.
        SENTRY_TRACES_SAMPLE_RATE: Sample rate for tracing (0.0 to 1.0). Default: 1.0
        SENTRY_SEND_DEFAULT_PII: Whether to send PII. Default: true
        SENTRY_ENVIRONMENT: Environment name (e.g., production, staging)
        SENTRY_RELEASE: Release/version string
        SENTRY_PROFILE_SESSION_SAMPLE_RATE: Profile session sample rate. Default: 1.0
        SENTRY_PROFILE_LIFECYCLE: Profile lifecycle mode. Default: trace
        SENTRY_ENABLE_LOGS: Enable Sentry logging integration. Default: true

    Returns:
        bool: True if Sentry was successfully initialized, False otherwise.

    Note:
        This function will silently return False if sentry-sdk is not installed
        or if SENTRY_DSN is not configured.
    """
    sentry_dsn = os.getenv("SENTRY_DSN")

    if not sentry_dsn:
        # Sentry is not configured - this is fine, it's optional
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.litellm import LiteLLMIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
    except ImportError:
        logger.warning(
            "Sentry SDK not installed but SENTRY_DSN is configured. "
            "To enable Sentry monitoring, install it with: pip install webex-bot-ai[sentry]"
        )
        return False

    try:
        # Parse configuration from environment variables
        traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0"))
        send_default_pii = _parse_bool(
            os.getenv("SENTRY_SEND_DEFAULT_PII"), default=True
        )
        environment: str | None = os.getenv("SENTRY_ENVIRONMENT")
        release: str | None = os.getenv("SENTRY_RELEASE")
        profile_session_sample_rate = float(
            os.getenv("SENTRY_PROFILE_SESSION_SAMPLE_RATE", "1.0")
        )
        profile_lifecycle = os.getenv("SENTRY_PROFILE_LIFECYCLE", "trace")
        enable_logs = _parse_bool(os.getenv("SENTRY_ENABLE_LOGS"), default=True)

        # Get package version for default release
        if not release:
            from importlib.metadata import PackageNotFoundError, version

            try:
                release = version("webex-bot-ai")
            except PackageNotFoundError:
                release = None

        # Build integrations list
        integrations = [
            LiteLLMIntegration(),
        ]

        # Add logging integration if enabled
        if enable_logs:
            integrations.append(
                LoggingIntegration(
                    level=logging.INFO,  # Capture info and above as breadcrumbs
                    event_level=logging.ERROR,  # Send errors as events
                )
            )

        # Initialize Sentry
        sentry_sdk.init(
            dsn=sentry_dsn,
            traces_sample_rate=traces_sample_rate,
            send_default_pii=send_default_pii,
            environment=environment,
            release=release,
            profile_session_sample_rate=profile_session_sample_rate,
            profile_lifecycle=profile_lifecycle,
            integrations=integrations,
        )

        logger.info(
            "Sentry monitoring enabled "
            "(traces_sample_rate=%.2f, environment=%s, release=%s)",
            traces_sample_rate,
            environment or "default",
            release or "unknown",
        )
        return True

    except Exception as e:
        logger.error("Failed to initialize Sentry: %s", e, exc_info=True)
        return False


def capture_exception(error: Exception) -> None:
    """Capture an exception to Sentry if available.

    Args:
        error: The exception to capture.
    """
    try:
        import sentry_sdk

        sentry_sdk.capture_exception(error)
    except ImportError:
        pass  # Sentry not installed, silently ignore


def set_user_context(user_id: str, email: str | None = None) -> None:
    """Set user context for Sentry events.

    Args:
        user_id: The unique user identifier.
        email: Optional user email.
    """
    try:
        import sentry_sdk

        sentry_sdk.set_user({"id": user_id, "email": email})
    except ImportError:
        pass  # Sentry not installed, silently ignore


def set_tag(key: str, value: str) -> None:
    """Set a tag for Sentry events.

    Args:
        key: The tag key.
        value: The tag value.
    """
    try:
        import sentry_sdk

        sentry_sdk.set_tag(key, value)
    except ImportError:
        pass  # Sentry not installed, silently ignore
