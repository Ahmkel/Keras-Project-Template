import os

from utils.sound import SoundUtils


def test_trim_file():
    in_path = os.path.join("uploads", "test1.wav")
    out_path = os.path.join("uploads", "out_test1.wav")
    SoundUtils.trim_file(in_path, out=out_path)
