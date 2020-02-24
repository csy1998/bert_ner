"""
@author: Shuyin Chen
@contact: shuyin_chen@shannonai.com

@version: 1.0
@file: set seed
@time: 2019/11/30 16:50
"""

from bert_ner.utils.logger import logger


class EvaluatorBase:
    """
    EvaluatorBase
    """
    def __init__(self,
                 params: dict):
        logger.info('Initializing Evaluator')
        self.params = params
        self.data_dir = params.data_dir
        self.multi_gpu = params.multi_gpu
        self.n_gpu = params.n_gpu
        self.is_master = params.is_master

        self.n_step = 0
        self.last_loss = 0
        self.total_loss = 0
        self.t_total = 0

    def eval(self):
        """
        eval model
        """
        # log eva info
        logger.info("***** Running evaluating *****")
        logger.info("  Num examples = %d", self.t_total)
        logger.info("  Num Epochs = 1")
        logger.info("  Instantaneous batch size per GPU = %d", self.params.eval_batch_size)
        logger.info("  Total eval batch size (w. parallel, distributed) = %d",
                    self.params.eval_batch_size * self.params.n_gpu)



