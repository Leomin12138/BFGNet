import glob
import os

from utils.dataset_processing import grasp, image
from utils.data.grasp_data import GraspDatasetBase
import numpy as np
import cv2

import matplotlib.pyplot as plt

class CornellDataset(GraspDatasetBase):
    """
    Dataset wrapper for the Cornell dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Cornell Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(CornellDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, '*', '*cpos.txt'))
        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

        self.depth_files = [f.replace('cpos.txt', 'd.tiff') for f in self.grasp_files]
        self.rgb_files = [f.replace('d.tiff', '.png') for f in self.depth_files]

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

if __name__ == "__main__":
    datasets = CornellDataset(file_path=r"/home/leo/code/二区/vmgnet/con")

    # 选择一个样本的索引
    idx = 12  # 可以根据需要修改索引

    # 获取数据
    depth_img = datasets.get_depth(idx=idx)
    print(depth_img.shape)
    rgb_img = datasets.get_rgb(idx=idx)
    print(rgb_img.shape)
    gtbb = datasets.get_gtbb(idx=idx)

    # 创建图像和坐标轴
    plt.figure(figsize=(12, 6))

    # 显示深度图
    plt.subplot(1, 2, 1)
    plt.imshow(depth_img, cmap='gray', vmin=0, vmax=1)  # 确保深度图的像素值范围在0到1之间
    plt.title("Depth Image")
    plt.axis("off")

    # 显示RGB图像与抓取框
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_img.transpose((1, 2, 0)))  # 转换为HWC格式
    plt.title("RGB Image with Grasp Rectangles")
    plt.axis("off")

    # 绘制每个抓取框
    for grasp_obj in gtbb.grs:
        # 获取中心点坐标（格式为(y, x)）
        y_center, x_center = grasp_obj.center
        angle = grasp_obj.angle  # 旋转角度（弧度）
        length = grasp_obj.length
        width = grasp_obj.width

        # 转换为处理后的图像中的(x, y)
        center = (x_center, y_center)

        # 计算旋转前的相对坐标
        half_l = length / 2
        half_w = width / 2
        points_rel = np.array([
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w]
        ])

        # 创建旋转矩阵（考虑图像坐标系的角度方向）
        rotation_matrix = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])

        # 应用旋转
        points_rot = np.dot(points_rel, rotation_matrix)


        # 平移到中心点
        points = points_rot + np.array(center)

        # 绘制抓取框
        plt.plot(
            points[[0, 1, 2, 3, 0], 0],  # X坐标
            points[[0, 1, 2, 3, 0], 1],  # Y坐标
            color='red', linewidth=2, linestyle='--'
        )

    plt.show()