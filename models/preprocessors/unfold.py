from typing import List, Any

from deeppavlov.core.common.registry import register


@register("unfold")
def Unfold(batch: List[List[Any]]) -> List[Any]:
    return [x for item in batch for x in item]
