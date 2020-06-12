# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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
import random
from logging import getLogger
from typing import Tuple, List

from bert_dp.tokenization import FullTokenizer

from deeppavlov.core.common.registry import register

from deeppavlov.models.preprocessors.bert_preprocessor import BertNerPreprocessor

log = getLogger(__name__)


@register('my_bert_ner_preprocessor')
class MyBertNerPreprocessor(BertNerPreprocessor):
    """Takes tokens and splits them into bert subtokens, encodes subtokens with their indices.
    Creates a mask of subtokens (one for the first subtoken, zero for the others).

    If tags are provided, calculates tags for subtokens.

    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        max_subword_length: replace token to <unk> if it's length is larger than this
            (defaults to None, which is equal to +infinity)
        token_masking_prob: probability of masking token while training
        provide_subword_tags: output tags for subwords or for words
        subword_mask_mode: subword to select inside word tokens, can be "first" or "last"
            (default="first")

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        max_subword_length: rmax lenght of a bert subtoken
        tokenizer: instance of Bert FullTokenizer
    """

    def _ner_bert_tokenize(self,
                           tokens: List[str],
                           tags: List[str],
                           tokenizer: FullTokenizer,
                           max_subword_len: int = None,
                           mode: str = None,
                           subword_mask_mode: str = "first",
                           token_masking_prob: float = None) -> Tuple[List[str], List[int], List[str]]:
        do_masking = (mode == 'train') and (token_masking_prob is not None)
        do_cutting = (max_subword_len is not None)
        tokens_subword = ['[CLS]']
        startofword_markers = [0]
        tags_subword = ['X']
        for token, tag in zip(tokens, tags):
            token_marker = int(tag != 'X')
            subwords = tokenizer.tokenize(token)
            if not subwords or (do_cutting and (len(subwords) > max_subword_len)):
                tokens_subword.append('[UNK]')
                startofword_markers.append(token_marker)
                tags_subword.append(tag)
            else:
                if do_masking and (random.random() < token_masking_prob):
                    tokens_subword.extend(['[MASK]'] * len(subwords))
                else:
                    tokens_subword.extend(subwords)
                if subword_mask_mode == "last":
                    startofword_markers.extend([0] * (len(subwords) - 1) + [token_marker])
                else:
                    startofword_markers.extend([token_marker] + [0] * (len(subwords) - 1))
                tags_subword.extend([tag] + ['X'] * (len(subwords) - 1))

        tokens_subword = tokens_subword[:self.max_seq_length - 1]
        startofword_markers = startofword_markers[:self.max_seq_length - 1]
        tags_subword = tags_subword[:self.max_seq_length - 1]
        tokens_subword.append('[SEP]')
        startofword_markers.append(0)
        tags_subword.append('X')
        return tokens_subword, startofword_markers, tags_subword
