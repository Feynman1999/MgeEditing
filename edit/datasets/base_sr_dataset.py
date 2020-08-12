import os.path as osp
import copy
from collections import defaultdict
from .base_dataset import BaseDataset
from pathlib import Path
from edit.utils import scandir, is_list_of


IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP')


class BaseSRDataset(BaseDataset):
    """Base class for super resolution Dataset.
    """

    def __init__(self, pipeline, scale, test_mode=False):
        super(BaseSRDataset, self).__init__(pipeline, test_mode)
        self.scale = scale

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
        results['scale'] = self.scale
        return self.pipeline(results)

    def evaluate(self, results):
        """Evaluate with different metrics.

        Args:
            results (list of dict): The output of forward_test() of the model.

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
