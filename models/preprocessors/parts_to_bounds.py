from typing import List, Tuple

from deeppavlov.core.common.registry import register


@register("parts_to_bounds")
def PartsToBounds(batch_text: List[str], batch_parts: List[List[str]]) -> List[List[Tuple[int, int]]]:
    batch_bounds = []
    for text, parts in zip(batch_text, batch_parts):
        bounds = []
        start = 0
        for part in parts:
            s = text.find(part, start)
            e = s + len(part)
            bounds.append((s, e))
            start = e
        batch_bounds.append(bounds)
    return batch_bounds
