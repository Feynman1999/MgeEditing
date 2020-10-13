import shutil
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import copy
from collections import defaultdict
from .base_dataset import BaseDataset
from pathlib import Path
from edit.utils import scandir, is_list_of, mkdir_or_exist, is_tuple_of, imread, imwrite


IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF')


class BaseMatchDataset(BaseDataset):
    """Base class for matching dataset.
    """

    def __init__(self, pipeline, mode="train"):
        super(BaseMatchDataset, self).__init__(pipeline, mode)
        pass

    @staticmethod
    def scan_folder(path):
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | :obj:`Path`): Folder path.

        Returns:
            list[str]: image list obtained form given folder.
        """

        if isinstance(path, (str, Path)):
            path = str(path)
        else:
            raise TypeError("'path' must be a str or a Path object, "
                            f'but received {type(path)}.')

        images = sorted(list(scandir(path, suffix=IMG_EXTENSIONS, recursive=True)))
        images = [osp.join(path, v) for v in images]
        assert images, f'{path} has no valid image file.'
        return images

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def evaluate(self, results, save_path):
        """Evaluate with different metrics.

        Args:
            results (list of dict): for every dict, record metric -> value for one frame

        Return:
            dict: Evaluation results dict.
        """
        assert is_list_of(results, dict), f'results must be a list of dict, but got {type(results)}'
        assert len(results) >= len(self), "results length should >= dataset length, due to multicard eval"
        self.logger.info("eval samples length: {}, dataset length: {}, only select front {} results".format(len(results), len(self), len(self)))
        results = results[:len(self)]

        eval_results = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                if "id" in metric:
                    continue
                eval_results[metric].append(val)

        for metric, val_list in eval_results.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        eval_results = {
            metric: sum(values) / len(self)
            for metric, values in eval_results.items()
        }

        return eval_results
