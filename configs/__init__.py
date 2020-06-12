from pathlib import Path

from deeppavlov.configs import Struct


def _build_configs_tree() -> Struct:
    root = Path(__file__).resolve().parent

    tree = {}

    for config in root.glob('**/*.json'):
        leaf = tree
        for part in config.relative_to(root).parent.parts:
            if part not in leaf:
                leaf[part] = {}
            leaf = leaf[part]
        leaf[config.stem] = config

    return Struct(tree)


configs = _build_configs_tree()
