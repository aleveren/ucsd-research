from nearby_peaks import nearby_peaks, nearby_peaks_slow

w_test = [10, 20, 30, 30, 40, 40, 50, 60]

def test_nearby_peaks():
    assert nearby_peaks(w_test, 34, delta = 9) == (2, 6)
    assert nearby_peaks(w_test, 34, delta = 5) == (2, 4)
    assert nearby_peaks(w_test, 34, delta = 3) == (4, 4)
    assert nearby_peaks(w_test, 25, delta = 7) == (1, 4)
    assert nearby_peaks(w_test, 45, delta = 7) == (4, 7)
    assert nearby_peaks(w_test, 55, delta = 7) == (6, 8)
    assert nearby_peaks(w_test, 35, delta = 100) == (0, 8)
    assert nearby_peaks(w_test, 5,  delta = 100) == (0, 8)
    assert nearby_peaks(w_test, 5,  delta = 3) == (0, 0)
    assert nearby_peaks(w_test, 65, delta = 3) == (8, 8)

def test_nearby_peaks_slow():
    assert nearby_peaks_slow(w_test, 34, delta = 9) == (2, 6)
    assert nearby_peaks_slow(w_test, 34, delta = 5) == (2, 4)
    assert nearby_peaks_slow(w_test, 34, delta = 3) == (4, 4)
    assert nearby_peaks_slow(w_test, 25, delta = 7) == (1, 4)
    assert nearby_peaks_slow(w_test, 45, delta = 7) == (4, 7)
    assert nearby_peaks_slow(w_test, 55, delta = 7) == (6, 8)
    assert nearby_peaks_slow(w_test, 35, delta = 100) == (0, 8)
    assert nearby_peaks_slow(w_test, 5,  delta = 100) == (0, 8)
    assert nearby_peaks_slow(w_test, 5,  delta = 3) == (0, 0)
    assert nearby_peaks_slow(w_test, 65, delta = 3) == (8, 8)
