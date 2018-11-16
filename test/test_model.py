from unittest import TestCase

from SegNet import SegNet
from inputs_object import read_images


class TestSegNet(TestCase):
    def test_model(self):
        segnet = SegNet()
        segnet.test()