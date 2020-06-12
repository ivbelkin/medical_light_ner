from logging import getLogger
from typing import List, Tuple

from deeppavlov.core.common.registry import register

log = getLogger(__name__)


@register("bounds_to_tags_bio")
def BoundsToTagsBio(batch_part_bounds: List[List[Tuple[int, int]]], batch_tag_bounds: List[List[Tuple[int, int, str]]]) -> List[List[str]]:
    batch_tags = []
    for part_bounds, tag_bounds in zip(batch_part_bounds, batch_tag_bounds):
        tags = []
        for part_s, part_e in part_bounds:
            t = 'O'
            for tag_s, tag_e, tag in tag_bounds:
                if tag_s <= part_s and part_e <= tag_e:
                    t = ('B' if part_s == tag_s else 'I') + '-' + tag
                    break
                elif tag_e <= part_s or part_e <= tag_s:
                    pass
                else:
                    log.error('Something went wrong! %i %i %i %i', part_s, part_e, tag_s, tag_e)
                    assert False
            if t.startswith('I-') and not (tags[-1].startswith('B-') or tags[-1].startswith('I-')):
                log.error('Something went wrong!')
                assert False
            tags.append(t)
        batch_tags.append(tags)
    return batch_tags
