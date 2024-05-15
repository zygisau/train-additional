import numpy as np
import torch
import numbers
from typing import List, Optional, Tuple, Union
from torchvision.transforms import functional as F


class ColorJitter(object):

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0,
    ) -> None:
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        else:
            raise TypeError(
                f"{name} should be a single number or a list/tuple with length 2.")

        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(
                f"{name} values should be between {bound}, but got {value}.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            return None
        else:
            return tuple(value)

    def __get_params(self,
                     brightness: Optional[List[float]],
                     contrast: Optional[List[float]],
                     saturation: Optional[List[float]],
                     hue: Optional[List[float]],
                     ) -> Tuple[torch.Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(
            torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(
            torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(
            torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(
            torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def _process_image(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        # fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.__get_params(
        #     self.brightness, self.contrast, self.saturation, self.hue
        # )
        brightness_factor = None if self.brightness is None else float(
            torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))

        img[[0, 1, 2], :, :] = F.adjust_brightness(img[[0, 1, 2], :, :], brightness_factor)
        # increase first three channels by brightness_factor
        # for fn_id in fn_idx:
            # if fn_id == 0 and brightness_factor is not None:
            #     img[[0, 1, 2], :, :] = F.adjust_brightness(
            #         img[[0, 1, 2], :, :], brightness_factor)
            # elif fn_id == 1 and contrast_factor is not None:
            #     img[[0, 1, 2], :, :] = F.adjust_contrast(
            #         img[[0, 1, 2], :, :], contrast_factor)
            # elif fn_id == 2 and saturation_factor is not None:
            #     img[[0, 1, 2], :, :] = F.adjust_saturation(
            #         img[[0, 1, 2], :, :], saturation_factor)
            # elif fn_id == 3 and hue_factor is not None:
            #     img[[0, 1, 2], :, :] = F.adjust_hue(
            #         img[[0, 1, 2], :, :], hue_factor)

        return img

    def __call__(self, samples):
        img1, img2, mask = samples

        img1 = self._process_image(img1)
        img2 = self._process_image(img2)
        return img1, img2, mask
