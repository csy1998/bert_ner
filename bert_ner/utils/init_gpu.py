"""
@author: Shuyin Chen
@contact: shuyin_chen@shannonai.com

@version: 1.0
@file: set seed
@time: 2019/11/30 14:50
"""

import os
import socket
import torch
from bert_ner.utils.logger import logger


def init_gpu_params(params):
    """
    单机多卡 gpu params
    Handle single and multi-GPU / multi-node.
    """
    if params.n_gpu <= 0:
        params.local_rank = 0
        params.master_port = -1
        params.is_master = True
        params.multi_gpu = False
        return

    assert torch.cuda.is_available()
    print("params.local_rank: ", params.local_rank)
    print("params.n_gpu: ", params.n_gpu)

    logger.info('Initializing GPUs')
    if params.n_gpu > 1:
        assert params.local_rank != -1
        params.multi_gpu = True

    # local job (single GPU)
    else:
        # 单卡 distributed 时 local_rank = 0
        # 单卡 不distributed 时 local_rank = -1
        # assert params.local_rank == -1
        params.local_rank = 0
        params.multi_gpu = False

    # sanity checks
    assert 0 <= params.local_rank < params.n_gpu
    params.is_master = params.local_rank == 0

    # summary
    logger.info("Local rank     : %i" % params.local_rank)
    logger.info("GPUs  :          %i" % params.n_gpu)
    logger.info("Master         : %s" % str(params.is_master))
    logger.info("Multi-GPU      : %s" % str(params.multi_gpu))
    logger.info("Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if params.multi_gpu:
        logger.info("Initializing PyTorch distributed")
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
        )