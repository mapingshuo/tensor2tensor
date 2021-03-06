# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

# This is far from being the real WMT17 task - only toyset here
# you need to register to get UN data and CWT data. Also, by convention,
# this is EN to ZH - use translate_enzh_wmt8k_rev for ZH to EN task
#
# News Commentary, around 220k lines
# This dataset is only a small fraction of full WMT17 task
_NC_TRAIN_DATASETS = [[
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12"
    ".tgz", [
        "training/news-commentary-v12.zh-en.en",
        "training/news-commentary-v12.zh-en.zh"
    ]
]]

# Test set from News Commentary. 2000 lines
_NC_TEST_DATASETS = [[
    "http://data.statmt.org/wmt17/translation-task/dev.tgz",
    ("dev/newsdev2017-enzh-src.en.sgm", "dev/newsdev2017-enzh-ref.zh.sgm")
]]

# UN parallel corpus. 15,886,041 lines
# Visit source website to download manually:
# https://conferences.unite.un.org/UNCorpus
#
# NOTE: You need to register to download dataset from official source
# place into tmp directory e.g. /tmp/t2t_datagen/dataset.tgz
_UN_TRAIN_DATASETS = [[
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/UNv1.0.en-zh.tar"
    ".gz", ["en-zh/UNv1.0.en-zh.en", "en-zh/UNv1.0.en-zh.zh"]
]]

_NIST_TRAIN_DATASET = [[
    "http://www.baidu.com",
    ["nist/en_zh_files/part-" + "%02d" % (i,) + ".en",
     "nist/en_zh_files/part-" + "%02d" % (i,) +".zh"]] \
    for i in range(64)
]

_NIST_TEST_DATASET = [[
    "http://www.baidu.com",
    ["nist_test/testset/nist06n.ref.frame.tok", # en
     "nist_test/testset/nist06n.src.tok"]] # zh
]

# CWMT corpus
# Visit source website to download manually:
# http://nlp.nju.edu.cn/cwmt-wmt/
#
# casia2015: 1,050,000 lines
# casict2015: 2,036,833 lines
# datum2015:  1,000,003 lines
# datum2017: 1,999,968 lines
# NEU2017:  2,000,000 lines
#
# NOTE: You need to register to download dataset from official source
# place into tmp directory e.g. /tmp/t2t_datagen/dataset.tgz

_CWMT_TRAIN_DATASETS = [[
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/casia2015/casia2015_en.txt", "cwmt/casia2015/casia2015_ch.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/casict2015/casict2015_en.txt", "cwmt/casict2015/casict2015_ch.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/neu2017/NEU_en.txt", "cwmt/neu2017/NEU_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2015/datum_en.txt", "cwmt/datum2015/datum_ch.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book1_en.txt", "cwmt/datum2017/Book1_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book2_en.txt", "cwmt/datum2017/Book2_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book3_en.txt", "cwmt/datum2017/Book3_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book4_en.txt", "cwmt/datum2017/Book4_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book5_en.txt", "cwmt/datum2017/Book5_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book6_en.txt", "cwmt/datum2017/Book6_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book7_en.txt", "cwmt/datum2017/Book7_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book8_en.txt", "cwmt/datum2017/Book8_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book9_en.txt", "cwmt/datum2017/Book9_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book10_en.txt", "cwmt/datum2017/Book10_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book11_en.txt", "cwmt/datum2017/Book11_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book12_en.txt", "cwmt/datum2017/Book12_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book13_en.txt", "cwmt/datum2017/Book13_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book14_en.txt", "cwmt/datum2017/Book14_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book15_en.txt", "cwmt/datum2017/Book15_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book16_en.txt", "cwmt/datum2017/Book16_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book17_en.txt", "cwmt/datum2017/Book17_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book18_en.txt", "cwmt/datum2017/Book18_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book19_en.txt", "cwmt/datum2017/Book19_cn.txt"]
], [
    "https://s3-us-west-2.amazonaws.com/twairball.wmt17.zh-en/cwmt.tgz",
    ["cwmt/datum2017/Book20_en.txt", "cwmt/datum2017/Book20_cn.txt"]
]]


def get_filename(dataset):
  return dataset[0][0].split("/")[-1]


@registry.register_problem
class TranslateEnzhWmt32k(translate.TranslateProblem):
  """Problem spec for WMT En-Zh translation.

  Attempts to use full training dataset, which needs website
  registration and downloaded manually from official sources:

  CWMT:
    - http://nlp.nju.edu.cn/cwmt-wmt/
    - Website contrains instructions for FTP server access.
    - You'll need to download CASIA, CASICT, DATUM2015, DATUM2017,
        NEU datasets

  UN Parallel Corpus:
    - https://conferences.unite.un.org/UNCorpus
    - You'll need to register your to download the dataset.

  NOTE: place into tmp directory e.g. /tmp/t2t_datagen/dataset.tgz
  """

  @property
  def approx_vocab_size(self):
    return 2**15  # 32k

  @property
  def source_vocab_name(self):
    return "vocab.enzh-en.%d" % self.approx_vocab_size

  @property
  def target_vocab_name(self):
    return "vocab.enzh-zh.%d" % self.approx_vocab_size

  def get_training_dataset(self, tmp_dir):
    """UN Parallel Corpus and CWMT Corpus need to be downloaded manually.

    Append to training dataset if available

    Args:
      tmp_dir: path to temporary dir with the data in it.

    Returns:
      paths
    """
    full_dataset = _NC_TRAIN_DATASETS
    for dataset in [_CWMT_TRAIN_DATASETS, _UN_TRAIN_DATASETS]:
      filename = get_filename(dataset)
      tmp_filepath = os.path.join(tmp_dir, filename)
      if tf.gfile.Exists(tmp_filepath):
        full_dataset += dataset
      else:
        tf.logging.info("[TranslateEzhWmt] dataset incomplete, you need to "
                        "manually download %s" % filename)
    return full_dataset

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    train_dataset = self.get_training_dataset(tmp_dir)
    datasets = train_dataset if train else _NC_TEST_DATASETS
    source_datasets = [[item[0], [item[1][0]]] for item in train_dataset]
    target_datasets = [[item[0], [item[1][1]]] for item in train_dataset]
    source_vocab = generator_utils.get_or_generate_vocab(
        data_dir,
        tmp_dir,
        self.source_vocab_name,
        self.approx_vocab_size,
        source_datasets,
        file_byte_budget=1e8)
    target_vocab = generator_utils.get_or_generate_vocab(
        data_dir,
        tmp_dir,
        self.target_vocab_name,
        self.approx_vocab_size,
        target_datasets,
        file_byte_budget=1e8)
    tag = "train" if train else "dev"
    filename_base = "wmt_enzh_%sk_tok_%s" % (self.approx_vocab_size, tag)
    data_path = translate.compile_data(tmp_dir, datasets, filename_base)
    return text_problems.text2text_generate_encoded(
        text_problems.text2text_txt_iterator(data_path + ".lang1",
                                             data_path + ".lang2"),
        source_vocab, target_vocab)

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }

@registry.register_problem
class TranslateEnzhNISTSmall(translate.TranslateProblem):
  """
  """

  @property
  def approx_vocab_size(self):
    return 2**15  # 32k

  @property
  def source_vocab_name(self):
    return "vocab.nist.enzh-en.%d" % self.approx_vocab_size

  @property
  def target_vocab_name(self):
    return "vocab.nist.enzh-zh.%d" % self.approx_vocab_size

  @property
  def dataset_splits(self):
    return [
        {
            "split": problem.DatasetSplit.TRAIN,
            "shards": 100,  # this is a small dataset
        },
        {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }
    ]

  def get_training_dataset(self, tmp_dir):
    """UN Parallel Corpus and CWMT Corpus need to be downloaded manually.

    Append to training dataset if available

    Args:
      tmp_dir: path to temporary dir with the data in it.

    Returns:
      paths
    """
    return _NIST_TRAIN_DATASET

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    train_dataset = self.get_training_dataset(tmp_dir)
    datasets = train_dataset if train else _NIST_TEST_DATASET
    source_datasets = [[item[0], [item[1][0]]] for item in train_dataset]
    target_datasets = [[item[0], [item[1][1]]] for item in train_dataset]

    # generate vocab file
    self.generate_vocab(tmp_dir,
                        source_datasets,
                        data_dir,
                        self.source_vocab_name,
                        self.approx_vocab_size)
    # generate vocab file
    self.generate_vocab(tmp_dir,
                        target_datasets,
                        data_dir,
                        self.target_vocab_name,
                        self.approx_vocab_size)

    source_vocab = generator_utils.get_or_generate_vocab(
        data_dir,
        tmp_dir,
        self.source_vocab_name,
        self.approx_vocab_size,
        source_datasets,
        file_byte_budget=1e8)

    target_vocab = generator_utils.get_or_generate_vocab(
        data_dir,
        tmp_dir,
        self.target_vocab_name,
        self.approx_vocab_size,
        target_datasets,
        file_byte_budget=1e8)

    tag = "train" if train else "dev"
    filename_base = "nist_enzh_%sk_tok_%s" % (self.approx_vocab_size, tag)
    data_path = translate.compile_data(tmp_dir, datasets, filename_base)
    return self.text2text_generate_encoded(
        text_problems.text2text_txt_iterator(data_path + ".lang1",
                                             data_path + ".lang2"),
        source_vocab, target_vocab)


  def generate_vocab(self, tmp_dir, datasets, data_dir, vocab_filename, approx_vocab_size):
    """
    split origin files with " ", then count, then write to vocab file like:"
    '<pad>'
    '<EOS>'
    'a'
    'b'
    "in utf-8
    """
    vocab_filepath = os.path.join(data_dir, vocab_filename)
    if vocab_filepath is not None and tf.gfile.Exists(vocab_filepath):
      tf.logging.info("Already found vocab file: %s, not creating", vocab_filepath)
      return None

    # generate vocab_file
    word_count = {}
    for dataset in datasets:
      filepath = os.path.join(tmp_dir, dataset[1][0])
      tf.logging.info("Reading original file %s for vocab file: %s", filepath, vocab_filepath)
      with tf.gfile.GFile(filepath, mode="r") as source_file:
        for line in source_file:
          words = line.strip().split(' ')
          for word in words:
            if word in word_count:  
                word_count[word] += 1
            else:   
                word_count[word] = 1
    word_count_tuple = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
    word_set = [x[0] for x in word_count_tuple][:approx_vocab_size-len(text_encoder.RESERVED_TOKENS)]
    word_set = text_encoder.RESERVED_TOKENS + word_set
    tf.logging.info("Generating vocab file: %s", vocab_filepath)
    with tf.gfile.Open(vocab_filepath, "w") as f:
      for word in word_set:
        f.write("'" + word + "'\n")
        #f.write("'" + word + "'\t" + str(word_count.get(word, 0)) + "\n")

  def text2text_generate_encoded(self,
                                 sample_generator,
                                 vocab,
                                 targets_vocab=None,
                                 has_inputs=True):
      """Encode Text2Text samples from the generator with the vocab."""
      targets_vocab = targets_vocab or vocab
      for sample in sample_generator:
          if has_inputs:
              sample["inputs"] = vocab.encode_without_subtoken(sample["inputs"])
              sample["inputs"].append(text_encoder.EOS_ID)
          sample["targets"] = targets_vocab.encode_without_subtoken(sample["targets"])
          sample["targets"].append(text_encoder.EOS_ID)
          # print(sample)
          yield sample

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }

@registry.register_problem
class TranslateEnzhWmt8k(TranslateEnzhWmt32k):
  """Problem spec for WMT En-Zh translation.

  This is far from being the real WMT17 task - only toyset here
  """

  @property
  def approx_vocab_size(self):
    return 2**13  # 8192

  @property
  def dataset_splits(self):
    return [
        {
            "split": problem.DatasetSplit.TRAIN,
            "shards": 10,  # this is a small dataset
        },
        {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }
    ]

  def get_training_dataset(self, tmp_dir):
    """Uses only News Commentary Dataset for training."""
    return _NC_TRAIN_DATASETS
