# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch RoBERTa model. """
import torch
import torch.utils.checkpoint
from torch import nn
import logging
from transformers import RobertaModel, AutoConfig

from src.models.modules.attention import LabelAttention


class PLMICD(nn.Module):
    def __init__(self, num_classes: int, model_path: str, **kwargs):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_path, num_labels=num_classes, finetuning_task=None
        )
        logging.info("init plmicd step 1")
        self.roberta = RobertaModel(self.config, add_pooling_layer=False)
        logging.info(model_path)
        self.roberta = self.roberta.from_pretrained(model_path, config=self.config) #STALLS OUT HERE
        logging.info("init plmicd step 3")
        self.attention = LabelAttention(
            input_size=self.config.hidden_size,
            projection_size=self.config.hidden_size,
            num_classes=num_classes,
        )
        self.loss = torch.nn.functional.binary_cross_entropy_with_logits
    # def __init__(self, config):
    #     super().__init__(config)
    #     self.num_labels = config.num_labels
    #     self.model_mode = config.model_mode
    #     self.roberta = RobertaModel(config, add_pooling_layer=False)
    #     self.dropout = nn.Dropout(config.hidden_dropout_prob)
    #     if "cls" in self.model_mode:
    #         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    #     elif "laat" in self.model_mode:
    #         self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
    #         self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
    #         self.third_linear = nn.Linear(config.hidden_size, config.num_labels)
    #     else:
    #         raise ValueError(f"model_mode {self.model_mode} not recognized")

    #     self.init_weights()

    def get_loss(self, logits, targets):
        return self.loss(logits, targets)

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        logits = self(data, attention_mask)
        loss = self.get_loss(logits, targets)
        logits = torch.sigmoid(logits)
        return {"logits": logits, "loss": loss, "targets": targets}

    def validation_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        logits = self(data, attention_mask)
        loss = self.get_loss(logits, targets)
        logits = torch.sigmoid(logits)
        return {"logits": logits, "loss": loss, "targets": targets}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        r"""
        input_ids (torch.LongTensor of shape (batch_size, num_chunks, chunk_size))
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_labels)`, `optional`):
        """

        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.roberta(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size)
            if attention_mask is not None
            else None,
            return_dict=False,
        )

        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        logits = self.attention(hidden_output)
        return logits
