from logging import getLogger
from typing import List, Tuple

from deeppavlov.core.common.registry import register

log = getLogger(__name__)


@register("relative_bounds")
def RelativeBounds(batch_part_bounds: List[List[Tuple[int, int]]], batch_tag_bounds: List[List[Tuple[int, int, str]]]) -> List[List[List[Tuple[int, int, str]]]]:
    batch_relative_tag_bounds = []
    for part_bounds, tag_bounds in zip(batch_part_bounds, batch_tag_bounds):
        relative_tag_bounds = []
        for part_s, part_e in part_bounds:
            part_relative_tag_bounds = []
            tag_bounds_rest = []
            for tag_s, tag_e, tag in tag_bounds:
                if part_s <= tag_s and tag_e <= part_e:
                    part_relative_tag_bounds.append((tag_s - part_s, tag_e - part_s, tag))
                elif tag_e <= part_s or part_e <= tag_s:
                    tag_bounds_rest.append((tag_s, tag_e, tag))
                else:
                    log.error('Something went wrong! %i %i %i %i', part_s, part_e, tag_s, tag_e)
                    assert False
            tag_bounds = tag_bounds_rest
            relative_tag_bounds.append(part_relative_tag_bounds)
        batch_relative_tag_bounds.append(relative_tag_bounds)
    return batch_relative_tag_bounds
