
from PIL import Image
from PIL import ImageEnhance
import PIL
import random
import numpy as np

class Brightness(object):
    def __init__(self, min=1, max=1) -> None:
        self.min = min
        self.max = max
    def __call__(self, clip):
        factor = random.uniform(self.min, self.max)
        if isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        new_clip = []
        for img in clip: 
            enh_bri = ImageEnhance.Brightness(img)
            new_img = enh_bri.enhance(factor=factor)
            new_clip.append(new_img)
        return new_clip

class Color(object):
    def __init__(self, min=1, max=1) -> None:
        self.min = min
        self.max = max
    def __call__(self, clip):
        factor = random.uniform(self.min, self.max)
        if isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        new_clip = []
        for img in clip: 
            enh_col = ImageEnhance.Color(img)
            new_img = enh_col.enhance(factor=factor)
            new_clip.append(new_img)
        return new_clip

class Contrast(object):
    def __init__(self, min=1, max=1) -> None:
        self.min = min
        self.max = max
    def __call__(self, clip):
        factor = random.uniform(self.min, self.max)
        if isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        new_clip = []
        for img in clip: 
            enh_con = ImageEnhance.Contrast(img)
            new_img = enh_con.enhance(factor=factor)
            new_clip.append(new_img)
        return new_clip

class Sharpness(object):
    def __init__(self, min=1, max=1) -> None:
        self.min = min
        self.max = max
    def __call__(self, clip):
        factor = random.uniform(self.min, self.max)
        if isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        new_clip = []
        for img in clip: 
            enh_sha = ImageEnhance.Sharpness(img)
            new_img = enh_sha.enhance(factor=1.5)
            new_clip.append(new_img)
        return new_clip