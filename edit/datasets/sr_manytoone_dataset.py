"""
    for many to one
"""
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS
from .utils import get_key_for_video_imgs
from edit.utils import scandir, is_list_of, mkdir_or_exist, is_tuple_of

IMG_EXTENSIONS = ('.png', )

@DATASETS.register_module()
class SRManyToOneDataset(BaseSRDataset):
    """Many to One dataset for video super resolution.

    The dataset loads several LQ (Low-Quality) frames and a center GT
    (Ground-Truth) frame. Then it applies specified transforms and finally
    returns a list containing paired data [images, label].

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        num_input_frames (int): Window size for input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        mode (str): "train", "test" or "eval"
    """

    def __init__(self,
                 lq_folder,
                 pipeline,
                 gt_folder = "",
                 num_input_frames = 7,
                 scale = 4,
                 mode = "train",
                 eval_part = None):  # for mge, we use ("08", "26")
        super(SRManyToOneDataset, self).__init__(pipeline, scale, mode)
        assert num_input_frames % 2 == 1, (
            f'num_input_frames should be odd numbers, '
            f'but received {num_input_frames }.')
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.num_input_frames = num_input_frames
        self.eval_part = eval_part
        if eval_part is not None:
            assert is_tuple_of(eval_part, str)
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        # get keys
        keys = sorted(list(scandir(self.lq_folder, suffix=IMG_EXTENSIONS, recursive=True)),
                        key=get_key_for_video_imgs)

        # do split for train and eval
        if self.eval_part is not None:
            if self.mode == "train":
                keys = [k for k in keys if k.split('/')[0] not in self.eval_part]
            elif self.mode == "eval":
                keys = [k for k in keys if k.split('/')[0] in self.eval_part]
            else:
                pass

        self.frame_num = dict()
        for key in keys:
            self.frame_num[key.split("/")[0]] = 0
        for key in keys:
            self.frame_num[key.split("/")[0]] += 1

        data_infos = []
        for key in keys:
            # do some checks, to make sure the key for LR and HR is same. 
            if self.mode in ("train", "eval"):
                gt_path = os.path.join(self.gt_folder, key)
                assert os.path.exists(gt_path), "please make sure the key for LR and HR is same"

            if self.mode == "train":
                data_infos.append(
                    dict(
                        lq_path=self.lq_folder,
                        gt_path=self.gt_folder,
                        key=key,
                        max_frame_num=self.frame_num[key.split("/")[0]],
                        num_input_frames=self.num_input_frames
                    )
                )
            elif self.mode == "eval":
                data_infos.append(
                    dict(
                        lq_path = self.lq_folder,
                        gt_path = self.gt_folder,
                        key = key,
                        max_frame_num=self.frame_num[key.split("/")[0]],
                        num_input_frames=self.num_input_frames
                    )
                )
            elif self.mode == "test":
                data_infos.append(
                    dict(
                        lq_path = self.lq_folder,
                        key = key,
                        max_frame_num=self.frame_num[key.split("/")[0]],
                        num_input_frames=self.num_input_frames
                    )
                )
            else:
                raise NotImplementedError("")

        return data_infos

    def evaluate(self, results, save_path):
        """Evaluate with different metrics.

        Args:
            results (list of dict): for every dict, record metric -> value for one frame

        Return:
            dict: Evaluation results dict.
        """
        save_path = os.path.join(save_path, "SVG")
        mkdir_or_exist(save_path)
        assert is_list_of(results, dict), f'results must be a list of dict, but got {type(results)}'
        assert len(results) >= len(self), "results length should >= dataset length, due to multicard eval"
        self.logger.info("eval samples length: {}, dataset length: {}, only select front {} results".format(len(results), len(self), len(self)))
        results = results[:len(self)]

        clip_names = sorted(self.frame_num.keys())  # e.g. [`city`, `walk`]
        frame_nums = [ self.frame_num[clip] for clip in clip_names ]

        eval_results = defaultdict(list)  # a dict of list
        
        do_frames = 0
        now_clip_idx = 0
        eval_results_one_clip = defaultdict(list)
        for res in results:
            for metric, val in res.items():
                eval_results_one_clip[metric].append(val)

            do_frames += 1
            if do_frames == frame_nums[now_clip_idx]:
                self.logger.info("{}: {} is ok".format(now_clip_idx, clip_names[now_clip_idx]))
                for metric, values in eval_results_one_clip.items():
                    # metric clip_names[now_clip_idx] values   to save an svg
                    average = sum(values) / len(values)
                    save_filename = clip_names[now_clip_idx] + "_" + metric 
                    title = "{} for {}, length: {}, average: {:.4f}".format(metric, clip_names[now_clip_idx], len(values), average)
                    plt.figure(figsize=(len(values) // 2 + 1, 8))
                    plt.plot(list(range(len(values))), values, label=metric)  # promise that <= 10000
                    plt.title(title)
                    plt.xlabel('frame idx')
                    plt.ylabel('{} value'.format(metric))
                    plt.legend()
                    fig = plt.gcf()
                    fig.savefig(os.path.join(save_path, save_filename + '.svg'), dpi=600, bbox_inches='tight')
                    # plt.show()
                    plt.clf()
                    plt.close()

                    eval_results[metric].append(average)

                do_frames = 0
                now_clip_idx += 1
                eval_results_one_clip = defaultdict(list)

        for metric, val_list in eval_results.items():
            assert len(val_list) == len(clip_names), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(clip_names)}')

        # average the results
        eval_results = {
            metric: sum(values) / len(values)
            for metric, values in eval_results.items()
        }

        return eval_results
