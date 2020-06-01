import os
from typing import Sequence, Tuple


def latest_model(foldername: str):
    model_names = os.listdir(foldername)
    prefix_less, prefix = remove_common_prefix(model_names)
    nums_only, suffix = remove_common_suffix(prefix_less)
    latest_num = max(int(n) for n in nums_only)
    return f"{prefix}{latest_num}{suffix}"


def remove_common_prefix(strs: Sequence[str]) -> Tuple[Sequence[str], str]:
    prefix = os.path.commonprefix(strs)
    return [s.replace(prefix, "") for s in strs], prefix


def remove_common_suffix(strs: Sequence[str]) -> Tuple[Sequence[str], str]:
    rev_strs = [reverse(s) for s in strs]
    prefix_less, prefix = remove_common_prefix(rev_strs)
    return [reverse(s) for s in prefix_less], reverse(prefix)


def reverse(s: str) -> str:
    return "".join(reversed(s))
