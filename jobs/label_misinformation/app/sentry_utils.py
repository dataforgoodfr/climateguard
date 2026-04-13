import os
import logging
import sentry_sdk
import tomli
from sentry_sdk.integrations.logging import LoggingIntegration

def get_app_version():
     # Open and read the pyproject.toml file
    with open('pyproject.toml', 'rb') as toml_file:
        pyproject_data = tomli.load(toml_file)

    # Access the version from the pyproject.toml file
    version = pyproject_data['project']['version']
    return version


def sentry_init():
    if os.environ.get("SENTRY_DSN", None) != None:
        logging.info("Sentry init")
        logging_kwargs = {}
        if os.getenv("SENTRY_LOGGING") == "true":
            logging_kwargs = dict(
                enable_logs=True,
                integrations=[
                    # Only send WARNING (and higher) logs to Sentry logs,
                    # even if the logger is set to a lower level.
                    LoggingIntegration(sentry_logs_level=logging.INFO),
                ]
            )
        sentry_sdk.init(
            enable_tracing=False,
            traces_sample_rate=0.3,
            # To set a uniform sample rate
            # Set profiles_sample_rate to 1.0 to profile 100%
            # of sampled transactions.
            # We recommend adjusting this value in production,
            profiles_sample_rate=0.3,
            release=get_app_version,
            # functions_to_trace=functions_to_trace,
            # integrations=[ # TODO : https://docs.sentry.io/platforms/python/integrations/ray/
            #     RayIntegration(),
            # ],
            **logging_kwargs
        )
    else:
        logging.info("Sentry not init - SENTRY_DSN not found")


def sentry_close():
    sentry_sdk.flush(timeout=2.0)
