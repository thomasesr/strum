"""Preprocessing module for audio separation and chart parsing."""

try:
    from src.preprocessing.pipeline import run_preprocessing
except ImportError:
    run_preprocessing = None

try:
    from src.preprocessing.separation import separate_stems
except ImportError:
    separate_stems = None

try:
    from src.preprocessing.clean_stems import preprocess_clean_stems
except ImportError:
    preprocess_clean_stems = None

try:
    from src.preprocessing.stem_extraction import StemsExtractor, check_extraction_tools
except ImportError:
    StemsExtractor = None
    check_extraction_tools = None

try:
    from src.preprocessing.karaoke import KaraokeConfig, separate_karaoke
except ImportError:
    KaraokeConfig = None
    separate_karaoke = None
