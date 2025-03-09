import os
import logging
import sentry_sdk

def sentry_init(version = "0.0"):
    if(os.environ.get("SENTRY_DSN", None) != None):
        logging.info("Sentry init")
        sentry_sdk.init(
            enable_tracing=False,
            traces_sample_rate=0.3,
            # To set a uniform sample rate
            # Set profiles_sample_rate to 1.0 to profile 100%
            # of sampled transactions.
            # We recommend adjusting this value in production,
            profiles_sample_rate=0.3,
            release=version,
            # functions_to_trace=functions_to_trace,
            # integrations=[ # TODO : https://docs.sentry.io/platforms/python/integrations/ray/
            #     RayIntegration(),
            # ],
        )
    else:
        logging.info("Sentry not init - SENTRY_DSN not found")


def sentry_close():
    sentry_sdk.flush(timeout=2.0)