from set_cover import set_cover_exact_slow, set_cover_approx_fast

# Usage: `py.test .`

example_sets = [
    [ 1, 2, 3, 4, 5, 6, 7],
    [ 8, 9,10,11,12,13,14],
    [ 1, 8],
    [ 2, 3, 9,10],
    [ 4, 5, 6, 7,11,12,13,14],
]

def test_set_cover_exact_slow():
    result = set_cover_exact_slow(example_sets)
    assert result == [0, 1]

    result = set_cover_exact_slow([[30], [20,30], [10]])
    assert result == [1, 2]

def test_set_cover_approx_fast():
    result = set_cover_approx_fast(example_sets)
    assert result == [4, 3, 2]

    result = set_cover_approx_fast([[30], [20,30], [10]])
    assert result == [1, 2]
