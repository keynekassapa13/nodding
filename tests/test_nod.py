import pytest
from nodding.nods import detect_nods

def test_detect_single_nod():
    pitch = [0, 3, 7, 10, 7, 3, 0]
    nod_mask, nod_indices = detect_nods(
        pitch,
        threshold=20.0,
        window_size=5,
        min_pause=1,
        pitch_threshold=1)
    assert True in nod_mask

def test_detect_multiple_nods():
    pitch = [0, 3, 7, 10, 7, 3, 0, 1, 2, 3, 4, 5]
    nod_mask, nod_indices = detect_nods(
        pitch,
        threshold=20.0,
        window_size=10,
        min_pause=1,
        pitch_threshold=1)
    assert True in nod_mask
    assert len(nod_indices) == 2
    
def test_no_nods():
    pitch = [0, 1, 2, 3, 4, 5]
    nod_mask, nod_indices = detect_nods(pitch)
    assert not any(nod_mask)
    assert len(nod_indices) == 0