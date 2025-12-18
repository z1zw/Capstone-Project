from app.clean.clean_connections import entropy


def test_entropy_zero():
    assert entropy([]) == 0.0


def test_entropy_positive():
    v = entropy([1, 1, 1, 1])
    assert v > 0
