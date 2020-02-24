# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: test_streamer
@time: 2019/11/19 17:47

    这一行开始写关于本文件的说明与解释
"""

import os
from tqdm import tqdm
from bert_ner.deploy.tagger import Tagger


def test_title_tagger():
    """test title tagger"""

    TITLE_TAGGER = Tagger(task="ner",
                          model_path="/data/nfsdata2/yuxian/train_logs/20191119_ner_w_title_2/",
                          batch_size=32,
                          cuda_devices=[2],
                          deploy_method="none")

    texts = [
        "北京大学学生李晓雅住在北京市的北京大学里。",
        "北大校长孙子军与中华人民共和国国务院总理李克强会面。",
        "售票员任翔远今天来香侬科技参观访问，前台王斐热烈欢迎。"
    ]

    goldens = [[{'term': '北京大学学生', 'tag': 'TITLE', 'begin': 0, 'end': 6},
                {'term': '李晓雅', 'tag': 'PER', 'begin': 6, 'end': 9},
                {'term': '北京市', 'tag': 'GPE', 'begin': 11, 'end': 14},
                {'term': '北京大学', 'tag': 'ORG', 'begin': 15, 'end': 19}],
               [{'term': '北大校长', 'tag': 'TITLE', 'begin': 0, 'end': 4},
                {'term': '孙子军', 'tag': 'PER', 'begin': 4, 'end': 7},
                {'term': '中华人民共和国国务院总理', 'tag': 'TITLE', 'begin': 8, 'end': 20},
                {'term': '李克强', 'tag': 'PER', 'begin': 20, 'end': 23}],
               [{'term': '售票员', 'tag': 'TITLE', 'begin': 0, 'end': 3},
                {'term': '任翔远', 'tag': 'PER', 'begin': 3, 'end': 6},
                {'term': '香侬科技', 'tag': 'ORG', 'begin': 9, 'end': 13},
                {'term': '前台', 'tag': 'TITLE', 'begin': 18, 'end': 20},
                {'term': '王斐', 'tag': 'PER', 'begin': 20, 'end': 22}]]

    outputs = TITLE_TAGGER.batch_ner(texts)
    assert str(outputs) == str(goldens)


def test_cws_tagger():
    """test cws tagger"""

    cws_tagger = Tagger(task="cws",
                        model_path="/data/nfsdata2/nlp_application/models/grammar-correction/chinese_dev_models/models/chinese_cws",
                        batch_size=32,
                        cuda_devices=[2],
                        deploy_method="none")

    texts = [
        "今天天气不错。",
    ]

    goldens = [[{'term': '今天', 'tag': 'W', 'begin': 0, 'end': 2},
                {'term': '天气', 'tag': 'W', 'begin': 2, 'end': 4},
                {'term': '不错', 'tag': 'W', 'begin': 4, 'end': 6},
                {'term': '。', 'tag': 'W', 'begin': 6, 'end': 7}]]

    outputs = cws_tagger.batch_ner(texts)
    assert str(outputs) == str(goldens)


def test_detect_tagger():
    """test detect tagger"""

    detect_tagger = Tagger(task="detect",
                           model_path="/data/nfsdata2/nlp_application/models/grammar-correction/chinese_dev_models/models/chinese_detect_old",
                           batch_size=64,
                           cuda_devices=[0],
                           deploy_method="none")

    texts = [
        "我爱背景天安门。",
        "天天向上",
    ]

    input_lines = []
    output_lines = []
    # input_path = "/data/nfsdata2/shuyin/data/baike.txt"
    # output_path = "/data/nfsdata2/shuyin/data/baike.gec"

    input_path = "/data/nfsdata2/shuyin/data/parallel_data/zhenfu.all"
    output_path = "/data/nfsdata2/shuyin/data/parallel_data/zhenfu.gec"

    with open(output_path, 'w') as output_f:
        output_f.write('\n')

    with open(input_path) as input_f:
        for line in input_f:
            line = ''.join(line.strip().split())
            input_lines.append(line)

    interval = 128
    total_cnt = 0
    t = int(len(input_lines) / interval)
    for i in tqdm(range(t+1)):
        outputs = detect_tagger.batch_ner(input_lines[i*interval : (i+1)*interval])
        output_lines += outputs
        tmp_cnt = 0
        with open(output_path, 'a') as output_f:
            for line in outputs:
                if '【*' in line and '*】' in line:
                    output_f.write(line.strip() + '\n')
                    tmp_cnt += 1
        total_cnt += tmp_cnt
        print("tmp_cnt: ", tmp_cnt)

    print("num all: ", total_cnt)


if __name__ == '__main__':
    test_detect_tagger()