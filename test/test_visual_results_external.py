from unittest import TestCase

from SegNet import SegNet
from inputs_object import read_images


class TestSegNet(TestCase):
    def test_visual_results_external_image(self):
        filenames = [
            '/var/local/data/skugele/COMP8150/project/images/test/image_0000283.png',
            '/var/local/data/skugele/COMP8150/project/images/test/image_0000019.png',
        ]

        images = read_images(filenames)
        segnet = SegNet()
        pred_tot, var_tot, inference_time = segnet.visual_results_external_image(images)