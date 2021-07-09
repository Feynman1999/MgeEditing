from collections import defaultdict
import matplotlib.pyplot as plt
import os.path as osp
from collections import defaultdict
from .base_dataset import BaseDataset
from pathlib import Path
from edit.utils import scandir, is_list_of, mkdir_or_exist, imread, imwrite


class BaseMotDataset(BaseDataset):
    def __init__(self, pipeline, mode="train"):
        super(BaseMotDataset, self).__init__(pipeline, mode)
    
    def evaluate(self, results, save_path):
        """ Evaluate with different metrics.
            Args:
                results (list of dict): for every dict, record metric -> value for one frame

            Return:
                dict: Evaluation results dict.
        """
        save_SVG_path = osp.join(save_path, "SVG")
        mkdir_or_exist(save_SVG_path)
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
            if do_frames == frame_nums[now_clip_idx]: # 处理一个clip
                clip_name = clip_names[now_clip_idx]
                self.logger.info("{}: {} is ok".format(now_clip_idx, clip_name))
                for metric, values in eval_results_one_clip.items():
                    # metric clip_name values   to save an svg
                    average = sum(values) / len(values)
                    save_filename = clip_name + "_" + metric 
                    title = "{} for {}, length: {}, average: {:.4f}".format(metric, clip_name, len(values), average)
                    plt.figure(figsize=(len(values) // 4 + 1, 8))
                    plt.plot(list(range(len(values))), values, label=metric)  # promise that <= 10000
                    plt.title(title)
                    plt.xlabel('frame idx')
                    plt.ylabel('{} value'.format(metric))
                    plt.legend()
                    fig = plt.gcf()
                    fig.savefig(osp.join(save_SVG_path, save_filename + '.svg'), dpi=600, bbox_inches='tight')
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
