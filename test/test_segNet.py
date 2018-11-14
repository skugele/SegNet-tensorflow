from unittest import TestCase

from SegNet import SegNet
from os import getcwd

class TestSegNet(TestCase):
    def test_visual_results(self):
        # print(getcwd())
        segnet = SegNet()
        segnet.visual_results('TRAIN', 1)

    def test_visual_results_external_image(self):
        pass
        # self.fail()
