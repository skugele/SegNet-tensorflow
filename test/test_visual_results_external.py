from unittest import TestCase

from SegNet import SegNet
from inputs_object import read_images


class TestSegNet(TestCase):
    def test_visual_results_external_image(self):
        filenames = [
            './external_images/DTU_Campus_Image1.jpg',
            './external_images/DTU_Campus_Image2.jpg',
            './external_images/DTU_Campus_Image3.jpg',
        ]

        images = read_images(filenames)
        segnet = SegNet()
        pred_tot, var_tot, inference_time = segnet.visual_results_external_image(images)