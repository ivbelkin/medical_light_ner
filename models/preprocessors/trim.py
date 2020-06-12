from typing import List, Any

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register("trim")
class Trim(Component):

    def __init__(self, max_lenght: int, **kwargs):
        self.max_lenght = max_lenght

    def __call__(self, batch: List[List[Any]]) -> List[List[Any]]:
        return [x[:self.max_lenght] for x in batch]
