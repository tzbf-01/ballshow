import glob
import re
import os.path as osp
from .bases import BaseImageDataset

class BallShow(BaseImageDataset):
    """
    自定义数据集加载类
    """
    # 这里对应 data/ 下的文件夹名称
    dataset_dir = 'BallShow'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(BallShow, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin

        # 处理文件夹
        # relabel=True 表示训练集会将 ID 重新映射为 0, 1, 2...
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print(f"=> BallShow loaded from {self.dataset_dir}")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        # 获取数据集统计信息
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """检查文件夹是否存在"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))  # 支持 .jpg
        # 如果有 .png 图片，可以追加: img_paths += glob.glob(osp.join(dir_path, '*.png'))

        # 正则表达式：匹配 "ID_cCamera_xxx" 格式
        # ([-\d]+) 匹配 ID (可能是负数)
        # c(\d) 匹配 摄像头编号
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            # 先遍历一遍获取所有 ID，用于 relabel
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue
            pid_container.add(pid)

        # 建立 ID 到 0~N 的映射
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())

            if pid == -1: continue  # 过滤垃圾图片

            # 摄像头编号通常从1开始，这里减1变成从0开始
            camid -= 1

            if relabel:
                pid = pid2label[pid]

            # 如果是测试集，不进行 relabel，直接用原始 PID (需加上偏移量 pid_begin)
            else:
                pid = self.pid_begin + pid

            # dataset 格式: (图片路径, ID, 摄像头ID, TrackID)
            # TrackID 暂时不用，固定为 1
            dataset.append((img_path, pid, camid, 1))

        return dataset