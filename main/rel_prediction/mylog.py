import logging
import time
import os

def log_creater(output_dir , diy_name=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if(diy_name==None):
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    else:
        log_name =diy_name+'.log'
    final_log_file = os.path.join(output_dir,log_name)


    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG) # set file log level

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.WARNING) # set outstream log level

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log



# my_logger=log_creater('./log')
# my_logger.info("info_test")
# my_logger.warning("warning_test")

