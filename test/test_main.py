import os


def test_generate_script_exists():
    assert os.path.exists("main.py")
