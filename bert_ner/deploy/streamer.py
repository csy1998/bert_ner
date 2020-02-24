# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: streamer
@time: 2019/11/19 17:15

    这一行开始写关于本文件的说明与解释
"""


import os
import torch
from service_streamer import ManagedModel
from bert_ner.models.bert_tagger import BertForSequenceLabeling


class ManagedLabelingModel(ManagedModel):
    """format the model to be used by gpu worker"""

    def init_model(self, model_path, max_length):
        """init model"""
        self.model = self.load_traced_model(model_path, max_length=max_length)
        self.max_length = max_length

    def predict(self, input_ids):
        """inference"""
        # todo(yuxian) 每次都to cuda应该很慢吧 =-=
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids).view([-1, self.max_length]).to("cuda")
        attention_mask = input_ids != 0
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        outputs = outputs[0].cpu().split(1)
        return outputs

    @staticmethod
    def load_traced_model(model_path: str, max_length=128):
        """load jit traced model"""
        jit_model_path = os.path.join(model_path, f"traced_model_len{max_length}.pt")
        using_jit_model = False
        if os.path.exists(jit_model_path):
            try:
                jit_model = torch.jit.load(jit_model_path).to("cuda")  # re-load
                jit_model.eval()
                using_jit_model = True
            except Exception as e:
                print(e)
        if not using_jit_model:
            model = BertForSequenceLabeling.from_pretrained(model_path, torchscript=True).to("cuda")
            model.eval()
            example_seq = torch.ones([1, max_length], dtype=torch.long).to("cuda")
            jit_model = torch.jit.trace(model, (example_seq, example_seq))
            torch.jit.save(jit_model, jit_model_path)
        return jit_model
