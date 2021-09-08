# -*- coding:utf-8 -*-
# @author Fenglongyu

import logging
import sys
import datetime

time = datetime.datetime.now()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
"""
level 的数值 和次序
级别        数值
CRITICAL    50
ERROR       40
WARNING     30
INFO        20
DEBUG       10
NOTSET      0
"""
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# StreamHandler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(level=logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# FileHandler
log_name = (time.year, time.month, time.day, time.hour, time.minute, time.second)
file_handler = logging.FileHandler('./logs/%s-%s-%s_%sh-%sm-%ss.log'%log_name)
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Log
if __name__ == "__main__":
    logger.info('This is a log info')
    logger.debug('Debugging')
    logger.warning('Warning exists')
    logger.info('Finish')
