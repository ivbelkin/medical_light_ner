import argparse
from pathlib import Path

from deeppavlov import build_model, train_model, evaluate_model
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.core.commands.train import read_data_by_config
from deeppavlov.download import deep_download
from deeppavlov.utils.pip_wrapper import install_from_config
from deeppavlov.core.common.file import find_config

import dataset_readers.medicine_light_reader
import models.preprocessors.bounds_to_tags_bio
import models.preprocessors.parts_to_bounds
import models.preprocessors.relative_bounds
import models.preprocessors.sent_tokenizer
import models.preprocessors.unfold
import models.bert.bert_sequence_tagger
import models.preprocessors.bert_preprocessor
import models.preprocessors.trim

from configs import configs


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="select a mode, train or interact", type=str,
                        choices={'train', 'download', 'install', 'evaluate'})
    parser.add_argument("config_path", help="path to a pipeline json config", type=str)
    return parser


def main(args):
    pipeline_config_path = Path(args.config_path).absolute()

    if args.mode == 'install':
        install_from_config(pipeline_config_path)
    elif args.mode == 'download':
        deep_download(pipeline_config_path)
    elif args.mode == 'train':
        train_model(pipeline_config_path, download=False)
    elif args.mode == 'evaluate':
        evaluate_model(pipeline_config_path, download=False)
    else:
        print('Mode', args.mode, 'not supported')

    # model = build_model(pipeline_config_path)

    # config_dict = parse_config(CONFIG)
    # data = read_data_by_config(config_dict)

    # print(model(['Дисфункция билиарного тракта на фоне деформации желчного пузыря.']))


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    main(args)
