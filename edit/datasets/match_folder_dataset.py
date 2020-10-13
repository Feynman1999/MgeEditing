import os.path as osp
import numpy as np
from .base_match_dataset import BaseMatchDataset
from .registry import DATASETS


@DATASETS.register_module()
class MatchFolderDataset(BaseMatchDataset):
    def __init__(self,
                 data_path,
                 opt_folder,  # folder
                 sar_folder,  # folder
                 file_list_name,  # txt file
                 pipeline,
                 mode='train',
                 scale = 2,
                 x_size = 800,
                 z_size = 512
                 ):
        super(MatchFolderDataset, self).__init__(pipeline, mode)
        self.scale = scale
        self.data_path = str(data_path)
        self.opt_folder = osp.join(self.data_path, opt_folder)
        self.sar_folder = osp.join(self.data_path, sar_folder)
        self.file_list = osp.join(self.data_path, file_list_name)
        self.x_size = x_size
        self.z_size = z_size

        self.data_infos = self.load_annotations()


    def load_annotations(self):
        data_infos = []
        # 获得所有的成对信息，从file_list_name中
        f = open(self.file_list, 'r')
        opt_list = []
        sar_list = []
        top_left_x = []
        top_left_y = []
        for line in f.readlines():
            if line.strip() == "":
                continue
            infos = line.strip().split(" ")
            if self.mode == "test":
                if len(infos) == 1:
                    continue
                assert len(infos) == 3
                opt_list.append(infos[2])
                sar_list.append(infos[1])
            else:
                assert len(infos) == 4
                opt_list.append(infos[0])
                sar_list.append(infos[1])
                top_left_x.append(float(infos[3])/self.scale)
                top_left_y.append(float(infos[2])/self.scale)
        f.close()

        for i in range(len(opt_list)):
            if self.mode == "train":
                assert opt_list[i][-4] == '.'
                data_infos.append(
                    dict(
                        opt_path=osp.join(self.opt_folder, opt_list[i]),
                        sar_path=osp.join(self.sar_folder, sar_list[i]),
                        bbox = np.array([top_left_x[i], 
                                         top_left_y[i], 
                                         top_left_x[i] + self.z_size/self.scale - 1, 
                                         top_left_y[i] + self.z_size/self.scale - 1]).astype(np.float32),
                        scale = self.scale,
                        class_id = int(opt_list[i][0]),
                        file_id = int(opt_list[i][:-4].split("_")[-1])
                    )
                )
            elif self.mode == "eval":
                data_infos.append(
                    dict(
                        opt_path=osp.join(self.opt_folder, opt_list[i]),
                        sar_path=osp.join(self.sar_folder, sar_list[i]),
                        bbox = np.array([top_left_x[i], 
                                         top_left_y[i], 
                                         top_left_x[i] + self.z_size/self.scale - 1, 
                                         top_left_y[i] + self.z_size/self.scale - 1]).astype(np.float32),
                        scale = self.scale,
                        class_id = int(opt_list[i][0]),
                        file_id = int(opt_list[i][:-4].split("_")[-1])
                    )
                )
            elif self.mode == "test":
                data_infos.append(
                    dict(
                        opt_path = osp.join(self.opt_folder, opt_list[i]),
                        sar_path = osp.join(self.sar_folder, sar_list[i]),
                        class_id = int(opt_list[i][0]),
                        file_id = int(opt_list[i][:-4].split("_")[-1])
                    )
                )
            else:
                raise NotImplementedError("not known mode: {}".format(self.mode))

        self.logger.info("MatchFolder dataset load ok, mode:{} len:{}".format(self.mode, len(data_infos)))
        return data_infos
