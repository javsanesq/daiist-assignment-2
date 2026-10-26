"""Windows entry point; training never runs when an app imports this module."""
from multiprocessing import freeze_support

from pipeline import run_pipeline


if __name__ == "__main__":
    freeze_support()
    run_pipeline()
