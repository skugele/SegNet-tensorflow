from unittest import TestCase

from SegNet import SegNet
from inputs_object import read_images


class TestSegNet(TestCase):
    def test_visual_results(self):
        segnet = SegNet()
        segnet.visual_results('TRAIN', [2])
