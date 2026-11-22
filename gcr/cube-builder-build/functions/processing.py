import numpy as np


def extract_qa_bits(band_data, bit_location):
    """Extrai o bit na posição informada dos dados de Quality Assessment.

    Args:
        band_data (numpy.ndarray): dados de QA.
        bit_location (int): posição do bit.
    """
    return band_data & (1 << bit_location)
