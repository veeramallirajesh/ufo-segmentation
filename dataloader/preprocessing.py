import cv2
import numpy as np
import torch
from albumentations import Resize
from PIL import Image
from .data_utils import DetectedObject


def strip_dict(dict1, keys_to_remove):
    additional_data = dict(dict1)
    for key in keys_to_remove:
        if key in additional_data:
            del additional_data[key]

    return additional_data


class ApplyPreprocessing:
    def __init__(self, preprocess_f):
        self.preprocess_f = preprocess_f

    def __call__(self, sample):
        image = np.asarray(sample.convert("RGB"))
        img = self.preprocess_f(image)
        img = Image.fromarray(image).convert('L')
        return img


class ToTensor:
    def __init__(self, require_org=True, target_type=np.float32, target_name='target'):
        self.require_org = require_org
        self.target_type = target_type
        self.target_name = target_name

    def __call__(self, sample):
        if self.require_org:
            image, target, original = sample['image'], sample[self.target_name], sample['original']
        else:
            image, target = sample['image'], sample[self.target_name]

        # move channels to the back only for the image (no channels in seg. map)
        # also for multiclass we have an int64 tensor for targets with indices
        img = np.moveaxis(image, -1, 0)

        # this is the last step in the chain, ignore everything unnecessary
        if self.require_org:
            return {'image': torch.from_numpy(img.astype(np.float32)),
                    self.target_name: torch.from_numpy(target.astype(self.target_type)),
                    'original': torch.from_numpy(original.astype(np.float32))}
        else:
            return {'image': torch.from_numpy(img.astype(np.float32)),
                    self.target_name: torch.from_numpy(target.astype(self.target_type))}


class Rescale:
    def __init__(self, output_size, output_size_y=None):
        assert isinstance(output_size, int)
        self.output_size_x = output_size
        self.output_size_y = output_size if output_size_y is None else output_size_y

    def __call__(self, sample):
        image, original, segmentation, objects = sample['image'], sample['original'],\
                                                 sample['segmentation'], sample['objects']

        new_obj = []
        scale_x = self.output_size_x / image.shape[0]
        scale_y = self.output_size_y / image.shape[1]

        for obj in objects:
            new_points = [[p[0] * scale_y, p[1] * scale_x] for p in obj.points]
            new_obj.append(DetectedObject(points=new_points, label=obj.label))

        img = Resize(self.output_size_x, self.output_size_y)(image=image)["image"]
        org = Resize(self.output_size_x, self.output_size_y)(image=original)["image"]
        seg = Resize(self.output_size_x, self.output_size_y, interpolation=cv2.INTER_NEAREST)(image=segmentation)["image"]
        seg = seg.astype(np.uint8)

        # apply rescale also to original and keypoints
        return {'image': img,
                'segmentation': seg,
                'original': org,
                'objects': new_obj,
                **strip_dict(sample, ['image', 'segmentation', 'original', 'objects'])}


class RescaleOnlyMask:
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample):
        segmentation = sample['segmentation']
        seg = Resize(self.output_size, self.output_size)(image=segmentation)["image"]

        return {'segmentation': seg,
                **strip_dict(sample, ['segmentation'])}
