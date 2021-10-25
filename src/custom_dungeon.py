from typing import Optional

from PIL import Image
from mapgen import Dungeon


class CustomDungeon(Dungeon):
    """
    Custom dungeon to wrap existed one with new logic.
    """

    def __init__(self, *args, reward_for_new: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.__reward_for_new = reward_for_new

    def step(self, action: int):
        observation, reward, done, info = super().step(action)
        if self.__reward_for_new is not None and info["is_new"]:
            reward += self.__reward_for_new
        return observation, reward, done, info

    def get_image_view(self, height: int = 500, width: int = 500) -> Image:
        return Image.fromarray(self._map.render(self._agent)).convert("RGB").resize((height, width), Image.NEAREST)
