import os


def test_generate_script_exists():
    assert os.path.exists("generate_dataset.py")
