from logging import getLogger
from pathlib import Path

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress

from nltk.tokenize import sent_tokenize

from models.preprocessors.sent_tokenizer import RuSentTokenizer
from models.preprocessors.unfold import Unfold
from models.preprocessors.parts_to_bounds import PartsToBounds
from models.preprocessors.relative_bounds import RelativeBounds

log = getLogger(__name__)


@register('medicine_light_reader')
class MedicineLightDatasetReader(DatasetReader):
    """Class to read training datasets in Medicine Light format"""

    def read(self,
             data_path: str,
             train_filelist: str,
             valid_filelist: str,
             test_filelist: str,
             tags: list,
             as_text: bool,
             skip_empty: bool):
        self.train_filelist = train_filelist
        self.valid_filelist = valid_filelist
        self.test_filelist = test_filelist
        self.tags = tags
        self.as_text = as_text

        data_path = Path(data_path)

        dataset = {}
        sent_tokenizer = RuSentTokenizer()

        for name in ['train', 'valid', 'test']:
            batch_text = []
            batch_tag_text_bounds = []
            filelist = getattr(self, name + '_filelist')
            with open(data_path/filelist, 'r') as f:
                paths = [line.strip() for line in f.readlines()]
            for path in paths:
                if path.endswith('.txt'):
                    path_txt = path
                    path_ann = path.replace('.txt', '.ann')
                    with open(path_txt, 'r') as f:
                        txt = f.read()
                    ann = self.parse_ann_file(path_ann)
                    batch_text.append(txt)
                    batch_tag_text_bounds.append(ann)

            if not as_text:
                batch_sents = sent_tokenizer(batch_text)
                batch_sent = Unfold(batch_sents)
                batch_sents_bounds = PartsToBounds(batch_text, batch_sents)
                batch_tag_sents_bounds = RelativeBounds(batch_sents_bounds, batch_tag_text_bounds)
                batch_tag_sent_bounds = Unfold(batch_tag_sents_bounds)
                dataset[name] = [(sent, tag_sent_bounds) 
                    for sent, tag_sent_bounds in zip(batch_sent, batch_tag_sent_bounds) 
                    if not skip_empty or len(tag_sent_bounds) > 0 or name != 'train']
            else:
                dataset[name] = [(text, tag_text_bounds) 
                    for text, tag_text_bounds in zip(batch_text, batch_tag_text_bounds)]

        return dataset

    def parse_ann_file(self, file_name: Path):
        tag_bounds = []
        with open(file_name, 'r') as f:
            for line in f:
                if line.startswith('T'):
                    _, tag, s, e, _ = line.split(maxsplit=4)
                    if tag in self.tags:
                        s, e = int(s), int(e)
                        tag_bounds.append((s, e, tag))
        tag_bounds = sorted(tag_bounds)
        return tag_bounds
