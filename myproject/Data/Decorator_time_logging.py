import logging
import time
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\Data\logging_time.txt",
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
    filemode="w"
)
def log_execution_time(function_to_measure):
    """

    :param function_to_measure: function that is called inside
    :return:  update file with logging information abuot function name time of execution and running time of function
    """
    @wraps(function_to_measure)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = function_to_measure(*args, **kwargs)
        end = time.time()

        # Calculate execution time
        time_execution = end - start
        if time_execution > 120:
            time_execution /= 60
            time_execution = f"{time_execution:.2f} minutes"
        else:
            time_execution = f"{time_execution:.2f} seconds"

        # Log execution time
        logging.info(f"Executed function: {function_to_measure.__name__} time:  {time_execution} ")

        return result

    return wrapper
