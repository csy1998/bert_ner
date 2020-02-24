# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: checks
@time: 2019/9/4 14:31

    这一行开始写关于本文件的说明与解释
"""


class ConfigurationError(Exception):
    """
    The exception raised by any Ifluent-Chinese object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)
