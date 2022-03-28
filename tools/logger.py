import logging

def create_log(level=20):
    logger = logging.getLogger()
    logger.setLevel(level)
    console_handler = get_console_handler()
    if not logger.handlers:
        logger.addHandler(console_handler)
    return logger


# get logging console handler
def get_console_handler(format=None):
    console_handler = logging.StreamHandler()
    if format is None:
        formatter = logging.Formatter('%(asctime)-15s - %(levelname)s - %(message)s')
    else:
        formatter = format
    console_handler.setFormatter(formatter)
    return console_handler


# get logging file handler
def get_file_handler(file, format=None):
    file_handler = logging.FileHandler(file)
    if format is None:
        formatter = logging.Formatter('%(asctime)-15s - %(levelname)s - %(message)s')
    else:
        formatter = format
    file_handler.setFormatter(formatter)
    return file_handler


if __name__ == '__main__':
    logger = create_log()
    logger.info("testing logger...")
