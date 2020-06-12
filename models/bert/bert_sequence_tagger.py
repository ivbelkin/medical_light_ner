# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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

from logging import getLogger
from typing import List, Union

import tensorflow as tf

from deeppavlov.core.common.registry import register

from deeppavlov.models.bert.bert_sequence_tagger import BertSequenceTagger, BertSequenceNetwork

log = getLogger(__name__)


@register('my_bert_sequence_tagger')
class MyBertSequenceTagger(BertSequenceTagger):
    """BERT-based model for text tagging. It predicts a label for every token (not subtoken) in the text.
    You can use it for sequence labeling tasks, such as morphological tagging or named entity recognition.
    See :class:`deeppavlov.models.bert.bert_sequence_tagger.BertSequenceNetwork`
    for the description of inherited parameters.

    Args:
        n_tags: number of distinct tags
        use_crf: whether to use CRF on top or not
        use_birnn: whether to use bidirection rnn after BERT layers.
            For NER and morphological tagging we usually set it to `False` as otherwise the model overfits
        birnn_cell_type: the type of Bidirectional RNN. Either `lstm` or `gru`
        birnn_hidden_size: number of hidden units in the BiRNN layer in each direction
        return_probas: set this to `True` if you need the probabilities instead of raw answers
    """

    def __init__(self,
                 n_tags: List[str],
                 keep_prob: float,
                 bert_config_file: str,
                 pretrained_bert: str = None,
                 attention_probs_keep_prob: float = None,
                 hidden_keep_prob: float = None,
                 use_crf=False,
                 encoder_layer_ids: List[int] = (-1,),
                 encoder_dropout: float = 0.0,
                 optimizer: str = None,
                 weight_decay_rate: float = 1e-6,
                 use_birnn: bool = False,
                 birnn_cell_type: str = 'lstm',
                 birnn_hidden_size: int = 128,
                 ema_decay: float = None,
                 ema_variables_on_cpu: bool = True,
                 return_probas: bool = False,
                 freeze_embeddings: bool = False,
                 freeze_encoder_up_to: int = 0,
                 learning_rate: float = 1e-3,
                 bert_learning_rate: float = 2e-5,
                 min_learning_rate: float = 1e-07,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: float = 1.0,
                #  train_max_seq_length: int = 512,
                 **kwargs) -> None:
        self.freeze_encoder_up_to = freeze_encoder_up_to
        # self.train_max_seq_length = train_max_seq_length
        super().__init__(n_tags=n_tags,
                         keep_prob=keep_prob,
                         bert_config_file=bert_config_file,
                         pretrained_bert=pretrained_bert,
                         attention_probs_keep_prob=attention_probs_keep_prob,
                         hidden_keep_prob=hidden_keep_prob,
                         use_crf=use_crf,
                         encoder_layer_ids=encoder_layer_ids,
                         encoder_dropout=encoder_dropout,
                         optimizer=optimizer,
                         weight_decay_rate=weight_decay_rate,
                         use_birnn=use_birnn,
                         birnn_cell_type=birnn_cell_type,
                         birnn_hidden_size=birnn_hidden_size,
                         ema_decay=ema_decay,
                         ema_variables_on_cpu=ema_variables_on_cpu,
                         return_probas=return_probas,
                         freeze_embeddings=freeze_embeddings,
                         learning_rate=learning_rate,
                         bert_learning_rate=bert_learning_rate,
                         min_learning_rate=min_learning_rate,
                         learning_rate_drop_div=learning_rate_drop_div,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         load_before_drop=load_before_drop,
                         clip_norm=clip_norm,
                         **kwargs)

    # def _build_feed_dict(self, input_ids, input_masks, y_masks, y=None):
    #     train = y is not None
    #     if train:
    #         input_ids = [x[:self.train_max_seq_length - 1] + x[-1] for x in input_ids]
    #         input_masks = [x[:self.train_max_seq_length - 1] + x[-1] for x in input_masks]
    #         y_masks = [x[:self.train_max_seq_length - 1] + x[-1] for x in y_masks]
    #         y = [x[:self.train_max_seq_length - 1] + x[-1] for x in y]
    #     feed_dict = self._build_basic_feed_dict(input_ids, input_masks, train=train)
    #     feed_dict[self.y_masks_ph] = y_masks
    #     if y is not None:
    #         feed_dict[self.y_ph] = y
    #     return feed_dict

    def get_train_op(self, loss: tf.Tensor, learning_rate: Union[tf.Tensor, float], **kwargs) -> tf.Operation:
        assert "learnable_scopes" not in kwargs, "learnable scopes unsupported"
    
        kwargs['learnable_scopes'] = ()
        bert_learning_rate = learning_rate * self.bert_learning_rate_multiplier
        train_ops = []

        # train_op for bert variables
        if not self.freeze_embeddings:
            kwargs['learnable_scopes'] = kwargs['learnable_scopes'] + ('bert/embeddings',)
        num_hidden_layers = self.bert_config.num_hidden_layers
        for layer_idx in range(num_hidden_layers + self.freeze_encoder_up_to, num_hidden_layers):
            kwargs['learnable_scopes'] = kwargs['learnable_scopes'] + ('bert/encoder/layer_%d' % layer_idx,)
        if len(kwargs['learnable_scopes']) > 0:
            bert_train_op = super(BertSequenceNetwork, self).get_train_op(loss,
                                                                          bert_learning_rate,
                                                                          **kwargs)
            train_ops.append(bert_train_op)

        # train_op for ner head variables
        kwargs['learnable_scopes'] = ('ner',)
        head_train_op = super(BertSequenceNetwork, self).get_train_op(loss,
                                                                      learning_rate,
                                                                      **kwargs)
        train_ops.append(head_train_op)

        return tf.group(*train_ops)
