# Copyright 2018 The Google AI Language Team Authors.
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
"""
This is the Data Loading Pipeline for Sentence Classifier Task adapted from:
    `https://github.com/google-research/bert/blob/master/run_classifier.py`
"""

import csv
import logging
import os,sys

import texar.torch as tx

class MyLogger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass

class InputExample:
    r"""A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        r"""Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence.
                For single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second
                sequence. Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures:
    r"""A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

#design your own data processing procdure
class DiyProcessor():
    def __init__(self,data_dir):
        self.data_dir = data_dir
    def get_data(self, mode):
        with open(os.path.join(self.data_dir,mode+".txt"),encoding = "utf8") as f:
            lines = f.readlines()
        lines = [each.strip().split("\t") for each in lines]
        return self._create_examples(lines,mode)

    def _create_examples(self,lines, set_type):
        r"""Creates examples for the training and dev sets."""
        examples = []
        if len(lines[1]) == 2 and len(lines[1][0]) == 1: # label  text_a
            for (i, line) in enumerate(lines):
                guid = f"{set_type}-{i}"
                text_a = tx.utils.compat_as_text(line[1])
                label = tx.utils.compat_as_text(line[0])
                examples.append(InputExample(guid=guid, text_a=text_a,
                                             text_b="", label=label))

        elif len(lines[1]) == 2 and len(lines[1][0]) > 1: # text_a  text_b  (when test file has no label)
            for (i, line) in enumerate(lines):
                guid = f"{set_type}-{i}"
                text_a = tx.utils.compat_as_text(line[0])
                text_b = tx.utils.compat_as_text(line[1])
                if set_type == "test":
                    label = "0"
                else:
                    print("the file is not for testing, yet contains no labels")
                    exit()
                examples.append(InputExample(guid=guid, text_a=text_a,
                                             text_b=text_b, label=label))
        elif len(lines[1]) == 1: # text_a (when test file has no label)
            for (i, line) in enumerate(lines):
                guid = f"{set_type}-{i}"
                text_a = tx.utils.compat_as_text(line[0])
                if set_type == "test":
                    label = "0"
                else:
                    print("the file is not for testing, yet contains no labels")
                    exit()
                examples.append(InputExample(guid=guid, text_a=text_a,
                                             text_b="", label=label))
        elif len(lines[1]) == 3: # label  text_a  text_b
            for (i, line) in enumerate(lines):
                guid = f"{set_type}-{i}"
                text_a = tx.utils.compat_as_text(line[1])
                text_b = tx.utils.compat_as_text(line[2])

                label = tx.utils.compat_as_text(line[0])
                examples.append(InputExample(guid=guid, text_a=text_a,
                                             text_b=text_b, label=label))
        return examples

    def get_labels(self,pos = 0): # pos means the index in one line of data in train.txt: [0, "this is an example."]
        with open(os.path.join(self.data_dir,"train.txt"),encoding = "utf8") as f:
            lines = f.readlines()
        labels = [each.strip().split("\t")[pos] for each in lines]
        labels = list(set(labels))
        return labels

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    r"""Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    input_ids, segment_ids, input_mask = \
        tokenizer.encode_text(text_a=example.text_a,
                              text_b=example.text_b,
                              max_seq_length=max_seq_length)

    label_id = label_map[example.label]

    # here we disable the verbose printing of the data
    if ex_index < 0:
        logging.info("*** Example ***")
        logging.info("guid: %s", example.guid)
        logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logging.info("input_ids length: %d", len(input_ids))
        logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        logging.info("label: %s (id = %d)", example.label, label_id)

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id)
    return feature


def convert_examples_to_features_and_output_to_files(
        examples, label_list, max_seq_length, tokenizer, output_file,
        feature_types):
    r"""Convert a set of `InputExample`s to a pickled file."""

    with tx.data.RecordData.writer(output_file, feature_types) as writer:
        for (ex_index, example) in enumerate(examples):
            feature = convert_single_example(ex_index, example, label_list,
                                             max_seq_length, tokenizer)

            features = {
                "input_ids": feature.input_ids,
                "input_mask": feature.input_mask,
                "segment_ids": feature.segment_ids,
                "label_ids": feature.label_id
            }
            writer.write(features)


def prepare_record_data(processor, tokenizer,
                        max_seq_length, output_dir,
                        feature_types):
    r"""Prepare record data.
    Args:
        processor: Data Preprocessor, which must have get_labels,
            get_train/dev/test/examples methods defined.
        tokenizer: The Sentence Tokenizer. Generally should be
            SentencePiece Model.
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        output_dir: The directory to save the pickled file in.
        feature_types: The original type of the feature.
    """
    label_list = processor.get_labels()

    train_examples = processor.get_data('train')
    train_file = os.path.join(output_dir, "train.pkl")
    convert_examples_to_features_and_output_to_files(
        train_examples, label_list, max_seq_length,
        tokenizer, train_file, feature_types)

    eval_examples = processor.get_data("dev")
    eval_file = os.path.join(output_dir, "eval.pkl")
    convert_examples_to_features_and_output_to_files(
        eval_examples, label_list,
        max_seq_length, tokenizer, eval_file, feature_types)

    test_examples = processor.get_data("test")
    test_file = os.path.join(output_dir, "predict.pkl")
    convert_examples_to_features_and_output_to_files(
        test_examples, label_list,
        max_seq_length, tokenizer, test_file, feature_types)
