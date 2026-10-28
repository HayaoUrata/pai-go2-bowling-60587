"""Japanese font lookup shared by video rendering scripts."""

from functools import lru_cache
from pathlib import Path
from PIL import ImageFont


@lru_cache(maxsize=10)
def font(size):
    for path in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if Path(path).exists():return ImageFont.truetype(path,size)
    raise RuntimeError("Japanese font required: install Noto Sans CJK or use macOS Arial Unicode.")
