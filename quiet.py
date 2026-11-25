import logging
import os
import warnings

if os.environ.get("VERBOSE") != "1":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    warnings.filterwarnings("ignore")

    logging.getLogger().setLevel(logging.ERROR)
    for name in (
        "umap",
        "pynndescent",
        "numba",
        "transformers",
        "sentence_transformers",
        "huggingface_hub",
        "torch",
        "sklearn",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
