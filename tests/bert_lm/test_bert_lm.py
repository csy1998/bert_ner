# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fei_wang@shannonai.com
@Time       : 2019/6
@Description:
"""

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

#from pytorch_transformers import DistilBertForMaskedLM, DistilBertConfig

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)


def run(model_path, test_path):

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
    # tokenizer = DistilBertTokenizer.from_pretrained(model_path)

    # Load pre-trained model (weights)
    # config = BertConfig.from_pretrained(model_path)
    # model = BertForMaskedLM(config)
    model = BertForMaskedLM.from_pretrained(model_path)
    # model = DistilBertForMaskedLM.from_pretrained(model_path)
    
    print(model)
    model.eval()
    # while True:
    with open(test_path, 'r') as f:
        for line in f:
            # raw_inputs = input("text, start, end").split()
            raw_inputs = line.split()
            raw_text = "".join(raw_inputs[:-2])

            # masked_index = int(raw_inputs[1])
            start = int(raw_inputs[-2])
            end = int(raw_inputs[-1])
            # mask_flag = raw_inputs[2]

            raw_tokens = ["[CLS]"]
            for c in raw_text:
                raw_tokens.append(c)
            raw_tokens.append("[SEP]")

            # if mask_flag == "t":
            #     raw_tokens[masked_index] = "[MASK]"
            for i in range(start, end):
                raw_tokens[i] = "[MASK]"

            text = " ".join(raw_tokens)
            print(text)

            tokenized_text = text.split()
            tokenized_text = [token if token in tokenizer.vocab else "[UNK]" for token in tokenized_text]
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

            # Create the segments tensors.
            segments_ids = [0] * len(tokenized_text)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            # Predict all tokens
            with torch.no_grad():
                predictions = model(tokens_tensor, segments_tensors)[0]
                predictions = torch.nn.functional.softmax(predictions, dim=2)

            for idx, masked_index in enumerate(range(start, end)):
                predicted_value_and_index = torch.sort(predictions[0, masked_index], descending=True)
                predicted_value = predicted_value_and_index[0].tolist()
                predicted_index = predicted_value_and_index[1].tolist()
                predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)

                cnt = 0
                print(idx, "========")
                for v, w in zip(predicted_value, predicted_token):
                    print(v, w)
                    cnt += 1
                    if cnt == 10:
                        break


if __name__ == '__main__':
    test_file_path = '/home/chenshuyin/tinybert/test/test_bert.txt'
    
    # model_path = "/data/nfsdata2/shuyin/model/bert_google/chinese_L-12_H-768_A-12"
    # model_path = "/data/nfsdata2/shuyin/model/bert_xunfei/chinese_roberta_wwm_ext_pytorch"
    # model_path = "/data/nfsdata2/shuyin/model/bert_brightmart/RoBERTa_zh_L12_PyTorch"
    # model_path = "/data/nfsdata2/shuyin/model/bert_brightmart/RoBERTa_zh_Large_PyTorch"
    bert_model_path = "/data/nfsdata2/shuyin/data/bert_distillation_large/tinybert_embd1_hidn1_attn1_pred1_t1/checkpoints_0"

    run(bert_model_path, test_file_path)
