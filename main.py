# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example of building a sentence classifier based on pre-trained BERT model.
"""

import argparse
import functools
import importlib
import logging
import os
from typing import Any

import torch
import torch.nn as nn

import torch.nn.functional as F
import texar.torch as tx

from utils import model_utils
from utils import data_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-model", default="config_model",
    help="Configuration of the downstream part of the model")
parser.add_argument(
    '--pretrained-model-name', type=str, default='bert-base-cased',
    choices=tx.modules.BERTEncoder.available_checkpoints(),
    help="Name of the pre-trained checkpoint to load.")
parser.add_argument(
    "--config-data", default="config_data", help="The dataset config.")
parser.add_argument(
    "--output-dir", default="models/",
    help="The output directory where the model checkpoints will be written.")
parser.add_argument(
    "--log_path", default="./logs/",
    help="path to store log information")
parser.add_argument(
    "--LMroot", default=r"D:\LM",
    help="path of local BERT checkpoint")
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path to a model checkpoint (including bert modules) to restore from.")
parser.add_argument(
    "--do-train", action="store_true",default=True,
    help="Whether to run training.")
parser.add_argument(
    "--do-eval", action="store_true",default=True,
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--do-test", action="store_true",default=True,
    help="Whether to run test on the test set.")
args = parser.parse_args()

sys.stdout = data_utils.MyLogger(args.log_path+"training_info.txt", sys.stdout)

config_data: Any = importlib.import_module(args.config_data)
config_model: Any = importlib.import_module(args.config_model)

config_data_dict = {
    k: v for k, v in config_data.__dict__.items()
    if not k.startswith('__')}
config_model_dict = {
    k: v for k, v in config_model.__dict__.items()
    if not k.startswith('__')}

print(config_data_dict)
print(config_model_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.root.setLevel(logging.INFO)

# DIY BERT for your personal tasks
class DiyBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = config_data.num_classes
        if self.num_classes ==2:
            self.is_binary = True
            self.cls_layer = nn.Linear(config_model.hidden_size, 1)
        else:
            self.is_binary = False
            self.cls_layer = nn.Linear(config_model.hidden_size, self.num_classes)
        self.bert =  tx.modules.encoders.bert_encoder.BERTEncoder(
            pretrained_model_name=args.pretrained_model_name,
            cache_dir = args.LMroot)
        self._dropout_layer = nn.Dropout(config_model.dropout)
    def forward(self,input_ids, input_length, segment_ids ):

        enc_outputs, pooled_cls_token = self.bert(input_ids,
                                       input_length,
                                       segment_ids
                                       )
        pooled_cls_token = self._dropout_layer(pooled_cls_token)
        pooled_cls_token = self.cls_layer(pooled_cls_token)
        if self.is_binary:
            preds = (torch.sigmoid(pooled_cls_token) > 0.5).long()
            pooled_cls_token = torch.flatten(pooled_cls_token)

        else:
            preds = torch.argmax(pooled_cls_token, dim=-1)
        preds = torch.flatten(preds)
        return pooled_cls_token,preds

def main() -> None:

    tx.utils.maybe_create_dir(args.output_dir)
    tx.utils.maybe_create_dir(args.log_path)

    num_train_data = config_data.num_train_data

    model = DiyBert()
    model.to(device)

    num_train_steps = int(num_train_data / config_data.train_batch_size *
                          config_data.max_train_epoch)
    num_warmup_steps = int(num_train_steps * config_data.warmup_proportion)
    logging.info("total training steps: %d",num_train_steps)
    logging.info("total warm up steps: %d",num_warmup_steps)
    # Builds learning rate decay scheduler


    vars_with_decay = []
    vars_without_decay = []
    for name, param in model.named_parameters():
        if 'layer_norm' in name or name.endswith('bias'):
            vars_without_decay.append(param)
        else:
            vars_with_decay.append(param)

    opt_params = [{
        'params': vars_with_decay,
        'weight_decay': 0.01,
    }, {
        'params': vars_without_decay,
        'weight_decay': 0.0,
    }]
    optim = tx.core.BertAdam(
        opt_params, betas=(0.9, 0.999), eps=1e-6, lr=config_model.static_lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, functools.partial(model_utils.get_lr_multiplier,
                                 total_steps=num_train_steps,
                                 warmup_steps=num_warmup_steps))

    train_dataset = tx.data.RecordData(hparams=config_data.train_hparam,
                                       device=device)
    eval_dataset = tx.data.RecordData(hparams=config_data.eval_hparam,
                                      device=device)
    test_dataset = tx.data.RecordData(hparams=config_data.test_hparam,
                                      device=device)

    iterator = tx.data.DataIterator(
        {"train": train_dataset, "eval": eval_dataset, "test": test_dataset}
    )

    def _compute_loss(logits, labels):
        if model.is_binary:
                loss = F.binary_cross_entropy_with_logits(     #sigmoid is included in binary_cross_entropy_with_logits
                logits.view(-1), labels.float().view(-1), reduction='mean')
        else:
            loss = F.cross_entropy( #F.cross_entropy combines log_softmax and nll_loss in a single function
                logits.view(-1, model.num_classes),
                labels.view(-1), reduction='mean')
        return loss

    def _train_epoch():
        iterator.switch_to_dataset("train")
        model.train()

        for batch in iterator:
            optim.zero_grad()
            input_ids = batch["input_ids"]
            segment_ids = batch["segment_ids"]
            labels = batch["label_ids"]
            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            logits, _ = model(input_ids, input_length, segment_ids)

            loss = _compute_loss(logits, labels)
            loss.backward()
            optim.step()
            scheduler.step()
            step = scheduler.last_epoch

            dis_steps = config_data.display_steps
            if dis_steps > 0 and step % dis_steps == 0:
                logging.info("step: %d / %d; loss: %f", step, num_train_steps, loss)

            eval_steps = config_data.eval_steps
            if eval_steps > 0 and step % eval_steps == 0:
                _eval_epoch()
                model.train()

    @torch.no_grad()
    def _eval_epoch():
        iterator.switch_to_dataset("eval")
        model.eval()

        nsamples = 0
        avg_rec = tx.utils.AverageRecorder()
        for batch in iterator:
            input_ids = batch["input_ids"]
            segment_ids = batch["segment_ids"]
            labels = batch["label_ids"]

            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            logits, preds = model(input_ids, input_length, segment_ids)

            loss = _compute_loss(logits, labels)
            accu = tx.evals.accuracy(labels, preds)
            batch_size = input_ids.size()[0]
            avg_rec.add([accu, loss], batch_size)
            nsamples += batch_size
        logging.info("eval accu: %.4f; loss: %.4f; nsamples: %d",
                     avg_rec.avg(0), avg_rec.avg(1), nsamples)

    @torch.no_grad()
    def _test_epoch():
        iterator.switch_to_dataset("test")
        model.eval()

        _all_preds = []
        for batch in iterator:
            input_ids = batch["input_ids"]
            segment_ids = batch["segment_ids"]

            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            _, preds = model(input_ids, input_length, segment_ids)

            _all_preds.extend(preds.tolist())

        output_file = os.path.join(args.output_dir, "test_results.tsv")
        with open(output_file, "w+") as writer:
            writer.write("\n".join(str(p) for p in _all_preds))
        logging.info("test output written to %s", output_file)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    if args.do_train:
        for _ in range(config_data.max_train_epoch):
            _train_epoch()
        states = {
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(states, os.path.join(args.output_dir, 'model.ckpt'))

    if args.do_eval:
        _eval_epoch()

    if args.do_test:
        _test_epoch()

if __name__ == "__main__":
    main()
