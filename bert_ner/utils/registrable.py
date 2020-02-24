# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: registrable
@time: 2019/9/4 14:13

    这一行开始写关于本文件的说明与解释
"""

from ifluent_chinese.utils.checks import ConfigurationError


class MetaRegistrable(type):
    """通过维护类内REGISTRY变量从而实现register的meta_class"""
    REGISTRY = {}

    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        if not name.endswith("Interface"):
            if name in meta.REGISTRY:
                message = f"Cannot register {name} as {cls.__name__}; name already in use for " \
                    f"{meta.REGISTRY[name].__name__}"
                raise ConfigurationError(message)
            meta.REGISTRY[name] = cls
        return cls
