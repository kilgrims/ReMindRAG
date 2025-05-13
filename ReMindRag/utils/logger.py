import logging

def setup_logger(name, level, log_file=None, time_format="none", console_output=False):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
        
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
        
        if time_format == "none":
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        elif time_format == "simple":
            formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
        else:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        if console_output:
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        if log_file:
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger

def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(5):
        self._log(5, message, args, **kwargs)