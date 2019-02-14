import logging


def get_logger(logger_name, log_file_name, level=logging.INFO):
    logger = logging.getLogger(logger_name)

    # checks to see whether this logger has any handlers configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    # setup file handler
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(level)

    # setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # setup format of logging
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class LoggerManager:
    _logger = None

    @staticmethod
    def get_logger(name, log_file=None):
        if LoggerManager._logger is None:
            if log_file is None:
                LoggerManager._logger = get_logger(name, '{}.log'.format(name))
            else:
                LoggerManager._logger = get_logger(name, log_file)
        else:
            assert isinstance(LoggerManager._logger, logging.Logger)
            return logging.getLogger('{}.{}'.format(LoggerManager._logger.name, name))

        return LoggerManager._logger

