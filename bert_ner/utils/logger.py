# encoding: utf-8
"""
@author: Shuyin Chen
@contact: shuyin_chen@shannonai.com

@version: 1.0
@file: logger set
@time: 2019/11/30 14:50
"""

import logging

# todo 选用其中一个logger即可
logging.basicConfig(format = '[%(asctime)s.%(msecs)03d][%(levelname)s]<%(name)s> %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


# PACKAGE_NAME = "bert_ner"
#
# def init_root_logger(root_name=PACKAGE_NAME):
#     # logger = logging.root
#     # use 'airtest' as root logger name to prevent changing other modules' logger
#     logger = logging.getLogger(root_name)
#     logger.setLevel(logging.INFO)
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter(
#         fmt='[%(asctime)s.%(msecs)03d][%(levelname)s]<%(name)s> %(message)s',
#         datefmt='%I:%M:%S'
#     )
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#
# init_root_logger(PACKAGE_NAME)
#
# def get_logger(name: str):
#     assert name.startswith(PACKAGE_NAME), f"logger name {name} should starts with {PACKAGE_NAME}"
#     logger = logging.getLogger(name)
#     return logger

# logger = get_logger('bert_ner_csy')
