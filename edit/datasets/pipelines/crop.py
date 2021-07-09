import random
import math
import os
import numpy as np
import cv2
from ..registry import PIPELINES
from edit.utils import imresize, imwrite
from skimage.segmentation import slic, mark_boundaries


@PIPELINES.register_module()
class MinimumBoundingBox_ByOpticalFlow(object):
    def __init__(self, blocksizes = [9, 8], n_segments = (50, 70), compactness = (10,20)):
        self.blocksizes = blocksizes
        self.n_segments = n_segments  # for skimage slic
        self.compactness = compactness  # for skimage slic
        self.flow_dir = "/work_base/datasets/REDS/train/train_sharp_bicubic/X4_RAFT_sintel"
        self.name_padding_len = 8
        self.max_pixel_num = 48**2
        self.threthld = 8*9
        self.scale = 4

    def check_valid(self, x):
        if x[0]<0 or x[0] >= 180:
            return False
        if x[1]<0 or x[1] >= 320:
            return False
        return True     

    def viz(self, mask, idx, tl, br, clipname, img):
        c = np.zeros((180,320))
        for h,w in mask:
            c[h,w]=1
        c = (c*255).astype(np.uint8)
        cv2.rectangle(c, (tl[1], tl[0]), (br[1], br[0]), (255,0,0), 1)
        cv2.rectangle(img, (tl[1], tl[0]), (br[1], br[0]), (255,0,0), 1)
        imwrite(c, "./{}/{}_mask.png".format(clipname, idx))
        imwrite(img, "./{}/{}_img.png".format(clipname, idx))

    def __call__(self, results):
        """
            crop lq and gt frames
            steps:
            1. for first frame of lq, do slic algorithm(compactness = 10, and n_segments uniformly select), and randomly select one area
            2. use optical flow information, crop along the motion trajectory (for lq frames) (different frames
            may be have different crop size, up to object spatial size, but all are integral multiple of the block size, for transformer easy training)
            the length may be reduce (because object spatial size is be too small)  (e.g. 30 -> 15)
            3. paired crop gt frames
            4. add meta information of location (for transformer position encoding)
        """
        clipname = os.path.dirname(results['LRkey'])
        lq_paths = results['lq_path']

        n_segments = random.randint(self.n_segments[0], self.n_segments[1])
        compactness = random.randint(self.compactness[0], self.compactness[1])
        segments_lq_first_frame = slic(results['lq'][0], n_segments=n_segments, compactness=compactness, start_label=0)
        # out1=mark_boundaries(results['lq'][0], segments_lq_first_frame)
        # imwrite((out1*255).astype(np.uint8), "./{}.png".format(clipname))
        
        # randomly select one area
        max_class_id = np.max(segments_lq_first_frame)
        select_class_id = random.randint(0, max_class_id)
        lq_masks = []
        mask_lq_first_frame = np.argwhere(select_class_id == segments_lq_first_frame)  # e.g. (672, 2) int64
        while mask_lq_first_frame.shape[0] > self.max_pixel_num:
            select_class_id = random.randint(0, max_class_id)
            mask_lq_first_frame = np.argwhere(select_class_id == segments_lq_first_frame)
        lq_masks.append(mask_lq_first_frame)

        first_frame_idx = os.path.basename(lq_paths[0])
        first_frame_idx = int(os.path.splitext(first_frame_idx)[0])
        for idx in range(first_frame_idx, first_frame_idx + len(lq_paths) - 1):
            # according    idx -> idx+1      flow   solve   idx+1  mask
            flowpath = os.path.join(self.flow_dir, clipname, "{}_{}.npy".format(str(idx).zfill(self.name_padding_len), str(idx+1).zfill(self.name_padding_len)))
            flow = np.load(flowpath)
            L = []
            for h,w in lq_masks[-1]:
                res = [int(flow[h,w,1]+0.5) + h, int(flow[h,w,0]+0.5) + w]
                if self.check_valid(res):
                    L.append(res)
            if len(L) < self.threthld:
                break
            new_mask = np.array(L)
            lq_masks.append(new_mask)

        # crop for lq and gt
        lq_crops = []
        gt_crops = []
        for idx in range(0, len(lq_masks)):
            tl = np.min(lq_masks[idx], axis=0) # top-left
            br = np.max(lq_masks[idx], axis=0) # bottom-right
            # make tl and br    are integral multiple of the block size
            tl = (np.floor(tl / self.blocksizes) * self.blocksizes ).astype(np.int64)
            br = (np.ceil((br+1) / self.blocksizes) * self.blocksizes ).astype(np.int64) - 1
            # viz
            # self.viz(lq_masks[idx], idx, tl, br, clipname, results['lq'][idx])
            length_h = br[0] - tl[0] + 1
            length_w = br[1] - tl[1] + 1
            lq_crops.append(results['lq'][idx][tl[0]:tl[0] + length_h, tl[1]:tl[1] + length_w, ...])
            gt_crops.append(results['gt'][idx][self.scale * tl[0]:self.scale * (tl[0] + length_h), self.scale*tl[1]: self.scale*(tl[1] + length_w), ...])
            # print(lq_crops[-1].shape, gt_crops[-1].shape)
        results['lq'] = lq_crops
        results['gt'] = gt_crops
        return results


@PIPELINES.register_module()
class Random_Crop_Opt_Sar(object):
    def __init__(self, keys, size, have_seed = False, Contrast=False):
        self.keys = keys
        self.size = size # 500, 320
        self.have_seed = have_seed
        self.Contrast = Contrast
        if self.Contrast:
            assert have_seed == False

    def get_optical_h_w(self, sar_h, sar_w):
        up = 800 - self.size[0]
        optical_h = random.randint(max(sar_h - (self.size[0]-self.size[1]), 0), min(sar_h, up))
        optical_w = random.randint(max(sar_w - (self.size[0]-self.size[1]), 0), min(sar_w, up))
        return optical_h, optical_w

    def __call__(self, results):
        if self.have_seed:  # 用于测试时
            random.seed(np.sum(results['sar']))

        gap = 512 - self.size[1]
        sar_h = random.randint(0, gap) # 随机两个数 去裁剪sar
        sar_w = random.randint(0, gap)
        # 获得sar图像
        results['sar'] = results['sar'][sar_h:sar_h+self.size[1], sar_w:sar_w+self.size[1], :]  # h,w,1
        # 所以我们可以得到裁剪出的sar图在800中的左上角
        sar_h = results['bbox'][0] + sar_h
        sar_w = results['bbox'][1] + sar_w
        
        if self.Contrast: # 随机三个用于训练
            sar = results['sar']
            optical = results['opt']
            results['sar'] = []
            results['opt'] = []
            results['bbox'] = []
            for _ in range(3):
                results['sar'].append(sar.copy())
                opt = optical.copy()
                optical_h, optical_w = self.get_optical_h_w(sar_h, sar_w)
                results['opt'].append(opt[optical_h:optical_h+self.size[0], optical_w:optical_w+self.size[0], :])
                results['bbox'].append(np.array([sar_h - optical_h, 
                                                 sar_w - optical_w, 
                                                 sar_h - optical_h + self.size[1] - 1, 
                                                 sar_w - optical_w + self.size[1] - 1]).astype(np.float32))
        else:
            optical_h, optical_w = self.get_optical_h_w(sar_h, sar_w)
            results['opt'] = results['opt'][optical_h:optical_h+self.size[0], optical_w:optical_w+self.size[0], :]  # h,w,1

            # 更改bbox
            results['bbox'][0] = sar_h - optical_h
            results['bbox'][1] = sar_w - optical_w
            results['bbox'][2] = results['bbox'][0] + self.size[1] - 1
            results['bbox'][3] = results['bbox'][1] + self.size[1] - 1
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys})')
        return repr_str


@PIPELINES.register_module()
class PairedRandomCrop(object):
    """Paried random crop.

    It crops a pair of lq and gt images with corresponding locations.
    It also supports accepting lq list and gt list.
    Required keys are "scale", "lq", and "gt",
    added or modified keys are "lq" and "gt".

    Args:
        gt_patch_size ([int, int]): cropped gt patch size.
    """

    def __init__(self, gt_patch_size, fix0=False, crop_flow=False):
        self.crop_flow = crop_flow
        if isinstance(gt_patch_size, int):
            self.gt_patch_h = gt_patch_size
            self.gt_patch_w = gt_patch_size
        else:
            self.gt_patch_h, self.gt_patch_w = gt_patch_size

        self.fix0 = fix0

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        scale = results['scale']
        assert self.gt_patch_h % scale == 0 and self.gt_patch_w % scale == 0
        lq_patch_h = self.gt_patch_h // scale
        lq_patch_w = self.gt_patch_w // scale

        lq_is_list = isinstance(results['lq'], list)
        if not lq_is_list:
            results['lq'] = [results['lq']]
        gt_is_list = isinstance(results['gt'], list)
        if not gt_is_list:
            results['gt'] = [results['gt']]

        h_lq, w_lq, _ = results['lq'][0].shape
        h_gt, w_gt, _ = results['gt'][0].shape

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise RuntimeError("HR's size is not {}X times to LR's size".format(scale))
            # do resize, resize gt to lq * scale
            # results['gt'] = [
            #     imresize(v, (w_lq * scale, h_lq * scale))
            #     for v in results['gt']
            # ]
            
        if h_lq < lq_patch_h or w_lq < lq_patch_w:
            raise ValueError(
                f'LQ ({h_lq}, {w_lq}) is smaller than patch size ',
                f'({lq_patch_h}, {lq_patch_w}). Please check '
                f'{results["lq_path"][0]} and {results["gt_path"][0]}.')

        # randomly choose top and left coordinates for lq patch
        if self.fix0:
            top = 0
            left = 0
        else:
            top = random.randint(0, h_lq - lq_patch_h)
            left = random.randint(0, w_lq - lq_patch_w)


        # crop lq patch
        results['lq'] = [
            v[top:top + lq_patch_h, left:left + lq_patch_w, ...]
            for v in results['lq']
        ]
        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        results['gt'] = [
            v[top_gt:top_gt + self.gt_patch_h,
              left_gt:left_gt + self.gt_patch_w, ...] for v in results['gt']
        ]

        # crop flow
        if self.crop_flow:
            # results['flow']    list of [h,w,2] 
            results['flow'] = [
                v[top:top + lq_patch_h, left:left + lq_patch_w, ...]
                for v in results['flow']
            ]

        if not lq_is_list:
            results['lq'] = results['lq'][0]
        if not gt_is_list:
            results['gt'] = results['gt'][0]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_patch_size={self.gt_patch_h}, {self.gt_patch_w})'
        return repr_str


@PIPELINES.register_module()
class RandomCrop(object):
    def __init__(self, keys, patch_size, fix0=False):
        if isinstance(patch_size, int):
            self.patch_h = patch_size
            self.patch_w = patch_size
        else:
            self.patch_h, self.patch_w = patch_size

        self.keys = keys
        self.fix0 = fix0

    def __call__(self, results):
        for key in self.keys:
            is_list = isinstance(results[key], list)
            if not is_list:
                results[key] = [results[key]]

            h, w, _ = results[key][0].shape
            
            # randomly choose top and left coordinates for patch
            if self.fix0:
                top = 0
                left = 0
            else:
                top = random.randint(0, h - self.patch_h)
                left = random.randint(0, w - self.patch_w)

            # crop lq patch
            results[key] = [
                v[top:top + self.patch_h, left:left + self.patch_w, ...]
                for v in results[key]
            ]

            if not is_list:
                results[key] = results[key][0]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(patch_size={self.patch_h}, {self.patch_w})'
        return repr_str


@PIPELINES.register_module()
class RandomCenterCropPadTwo(object):
    """Random center crop and random around padding for CornerNet.
    This operation generates randomly cropped image from the original image and
    pads it simultaneously. Different from :class:`RandomCrop`, the output
    shape may not equal to ``crop_size`` strictly. We choose a random value
    from ``ratios`` and the output shape could be larger or smaller than
    ``crop_size``. The padding operation is also different from :class:`Pad`,
    here we use around padding instead of right-bottom padding.
    The relation between output image (padding image) and original image:
    .. code:: text
                        output image
               +----------------------------+
               |          padded area       |
        +------|----------------------------|----------+
        |      |         cropped area       |          |
        |      |         +---------------+  |          |
        |      |         |    .   center |  |          | original image
        |      |         |        range  |  |          |
        |      |         +---------------+  |          |
        +------|----------------------------|----------+
               |          padded area       |
               +----------------------------+
    There are 5 main areas in the figure:
    - output image: output image of this operation, also called padding
      image in following instruction.
    - original image: input image of this operation.
    - padded area: non-intersect area of output image and original image.
    - cropped area: the overlap of output image and original image.
    - center range: a smaller area where random center chosen from.
      center range is computed by ``border`` and original image's shape
      to avoid our random center is too close to original image's border.
    the summary pipeline is listed below.
    Train pipeline:
    1. Choose a ``random_ratio`` from ``ratios``, the shape of padding image
       will be ``random_ratio * crop_size``.
    2. Choose a ``random_center`` in center range.
    3. Generate padding image with center matches the ``random_center``.
    4. Initialize the padding image with pixel value equals to ``mean``.
    5. Copy the cropped area to padding image.
    6. Refine annotations.
    Args:
        ratios (tuple): random select a ratio from tuple and crop image to
            (crop_size[0] * ratio) * (crop_size[1] * ratio).
            Only available in train mode.
        border (int): max distance from center select area to image border.
            Only available in train mode.
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
    """
    def __init__(self,
                 ratios=(0.9, 1.0, 1.1),
                 border=128,
                 mean=None,
                 std=None,
                 to_rgb=None,
                 bbox_clip_border=True):
        assert isinstance(ratios, (list, tuple))
        self.ratios = ratios
        self.border = border
        # We do not set default value to mean, std and to_rgb because these
        # hyper-parameters are easy to forget but could affect the performance.
        # Please use the same setting as Normalize for performance assurance.
        assert mean is not None and std is not None and to_rgb is not None
        self.to_rgb = to_rgb
        self.input_mean = mean
        self.input_std = std
        if to_rgb:
            self.mean = mean[::-1]
            self.std = std[::-1]
        else:
            self.mean = mean
            self.std = std
        self.bbox_clip_border = bbox_clip_border

    def _get_border(self, border, size):
        """Get final border for the target size.
        This function generates a ``final_border`` according to image's shape.
        The area between ``final_border`` and ``size - final_border`` is the
        ``center range``. We randomly choose center from the ``center range``
        to avoid our random center is too close to original image's border.
        Also ``center range`` should be larger than 0.
        Args:
            border (int): The initial border, default is 128.
            size (int): The width or height of original image.
        Returns:
            int: The final border.
        """
        k = 2 * border / size
        i = pow(2, np.ceil(np.log2(np.ceil(k))) + (k == int(k)))
        return border // i

    def _filter_boxes(self, patch, boxes):
        """Check whether the center of each box is in the patch.
        Args:
            patch (list[int]): The cropped area, [left, top, right, bottom].
            boxes (numpy array, (N x 4)): Ground truth boxes.
        Returns:
            mask (numpy array, (N,)): Each box is inside or outside the patch.
        """
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = (center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) * (
            center[:, 0] < patch[2]) * (
                center[:, 1] < patch[3])
        return mask

    def _crop_image_and_paste(self, image, center, size):
        """Crop image with a given center and size, then paste the cropped
        image to a blank image with two centers align.
        This function is equivalent to generating a blank image with ``size``
        as its shape. Then cover it on the original image with two centers (
        the center of blank image and the random center of original image)
        aligned. The overlap area is paste from the original image and the
        outside area is filled with ``mean pixel``.
        Args:
            image (np array, H x W x C): Original image.
            center (list[int]): Target crop center coord.
            size (list[int]): Target crop size. [target_h, target_w]
        Returns:
            cropped_img (np array, target_h x target_w x C): Cropped image.
            border (np array, 4): The distance of four border of
                ``cropped_img`` to the original image area, [top, bottom,
                left, right]
            patch (list[int]): The cropped area, [left, top, right, bottom].
        """
        center_y, center_x = center
        target_h, target_w = size
        img_h, img_w, img_c = image.shape

        x0 = max(0, center_x - target_w // 2)
        x1 = min(center_x + target_w // 2, img_w)
        y0 = max(0, center_y - target_h // 2)
        y1 = min(center_y + target_h // 2, img_h)
        patch = np.array((int(x0), int(y0), int(x1), int(y1)))

        left, right = center_x - x0, x1 - center_x
        top, bottom = center_y - y0, y1 - center_y

        cropped_center_y, cropped_center_x = target_h // 2, target_w // 2
        cropped_img = np.zeros((target_h, target_w, img_c), dtype=image.dtype)
        for i in range(img_c):
            cropped_img[:, :, i] += self.mean[i]
        y_slice = slice(cropped_center_y - top, cropped_center_y + bottom)
        x_slice = slice(cropped_center_x - left, cropped_center_x + right)
        cropped_img[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

        border = np.array([
            cropped_center_y - top, cropped_center_y + bottom,
            cropped_center_x - left, cropped_center_x + right
        ],dtype=np.float32)

        return cropped_img, border, patch

    def _train_aug(self, results, scale):
        """Random crop and around padding the original image.
        Args:
            results (dict): Image infomations in the augment pipeline.
        Returns:
            results (dict): The updated dict.
        """
        img1 = results['img1']
        h, w, c = img1.shape
        img2 = results['img2']
        h2, w2, c = img2.shape
        assert h==h2 and w == w2

        boxes1 = results['gt_bboxes1']
        boxes2 = results['gt_bboxes2']
        new_h = int(h * scale)
        new_w = int(w * scale)
        h_border = self._get_border(self.border, h)
        w_border = self._get_border(self.border, w)
        for i in range(30):
            center_x = np.random.randint(low=w_border, high=w - w_border)
            center_y = np.random.randint(low=h_border, high=h - h_border)

            cropped_img1, border, patch1 = self._crop_image_and_paste(img1, [center_y, center_x], [new_h, new_w])
            cropped_img2, border, patch2 = self._crop_image_and_paste(img2, [center_y, center_x], [new_h, new_w])

            mask1 = self._filter_boxes(patch1, boxes1)
            mask2 = self._filter_boxes(patch2, boxes2)

            # if image do not have valid bbox, any crop patch is valid.
            if not mask1.any() and len(boxes1) > 0:
                continue
            if not mask2.any() and len(boxes2) > 0:
                continue

            results['img1'] = cropped_img1
            results['img2'] = cropped_img2            
            results['img_shape'] = cropped_img1.shape
            results['pad_shape'] = cropped_img1.shape

            cropped_center_x, cropped_center_y = new_w // 2, new_h // 2

            # crop bboxes accordingly and clip to the image boundary
            for key in ['gt_bboxes1', 'gt_bboxes2']:
                mask = self._filter_boxes(patch1, results[key])
                bboxes = results[key][mask]
                bboxes[:, 0:4:2] += cropped_center_x - center_x
                bboxes[:, 1:4:2] += cropped_center_y - center_y
                if self.bbox_clip_border:
                    bboxes[:, 0:4:2] = np.clip(bboxes[:, 0:4:2], 0, new_w-1)
                    bboxes[:, 1:4:2] = np.clip(bboxes[:, 1:4:2], 0, new_h-1)
                keep = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
                bboxes = bboxes[keep]
                results[key] = bboxes
                if '1' in key:
                    labels = results['gt_labels1'][mask]
                    labels = labels[keep]
                    results['gt_labels1'] = labels
                else:
                    labels = results['gt_labels2'][mask]
                    labels = labels[keep]
                    results['gt_labels2'] = labels

            return results
        raise RuntimeError("can not find one crop in 30 times random")

    def __call__(self, results):
        img = results['img1']
        assert img.dtype == np.float32, (
            'RandomCenterCropPad needs the input image of dtype np.float32,'
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline')
        h, w, c = img.shape
        assert c == len(self.mean)
        scale = np.random.choice(self.ratios)
        return self._train_aug(results, scale)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'ratios={self.ratios}, '
        repr_str += f'border={self.border}, '
        repr_str += f'mean={self.input_mean}, '
        repr_str += f'std={self.input_std}, '
        repr_str += f'to_rgb={self.to_rgb}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str
