# some basic global variables
import logging
import time
import requests

# some basic global variables
API_KEY: str = "mZDI2HhRfp4QDHQ5KxGG3vvzwUpVUQRm"
BASE_URL_v3: str = "https://financialmodelingprep.com/api/v3/"
BASE_URL_v4: str = "https://financialmodelingprep.com/api/v4/"


# some basic url functions
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 30
import time


def rate_limits(limit_per_min):
    def decorator(func):
        last_call = 0

        def wrapper(*args, **kwargs):
            nonlocal last_call
            # Calculate time elapsed since last reset
            now = time.time()
            elapsed = now - last_call
            if elapsed < 60 / limit_per_min:
                time.sleep(60 / limit_per_min - elapsed)
            last_call = now
            # Call the original function
            return func(*args, **kwargs)

        return wrapper

    return decorator


@rate_limits(300)
def __api_access(url):
    not_finish = True
    return_var = []
    retry = 0
    while not_finish:
        try:
            response = requests.get(
                url=url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )
            if len(response.content) > 0:
                return_var = response.json()

            if len(response.content) == 0 or (
                    isinstance(return_var, dict) and len(return_var.keys()) == 0
            ):
                logging.warning("Response appears to have no data.  Returning empty List.")
            not_finish = False
        except requests.Timeout:
            time.sleep(0.01)
            retry += 1
            if retry > 5:
                not_finish = False
                logging.error(f"Connection to {url} timed out.")
            else:
                not_finish = True
        except requests.ConnectionError:
            time.sleep(0.01)
            retry += 1
            if retry > 5:
                not_finish = False
                logging.error(
                    f"Connection to {url} failed:  DNS failure, refused connection or some other connection related "
                    f"issue."
                )
            else:
                not_finish = True
        except requests.TooManyRedirects:
            logging.error(
                f"Request to {url} exceeds the maximum number of predefined redirections."
            )
            not_finish = False
        except Exception as e:
            logging.error(
                f"A requests exception has occurred that we have not yet detailed an 'except' clause for.  "
                f"Error: {e}"
            )
            not_finish = False

    return return_var