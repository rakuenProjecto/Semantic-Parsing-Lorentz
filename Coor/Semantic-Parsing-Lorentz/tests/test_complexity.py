from src.complexity import compute_linguistic_complexity


def test_complex_sentence_scores_higher_than_simple_sentence() -> None:
    simple = compute_linguistic_complexity("Book a flight.")
    complex_sentence = compute_linguistic_complexity(
        "Book a cheap direct flight from Seoul to Boston after Monday unless the calendar "
        "conflicts with the meeting."
    )

    assert complex_sentence.complexity_1_to_4 > simple.complexity_1_to_4
    assert 1.0 <= simple.complexity_1_to_4 <= 4.0
    assert 1.0 <= complex_sentence.complexity_1_to_4 <= 4.0
