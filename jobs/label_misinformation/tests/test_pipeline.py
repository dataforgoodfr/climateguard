from jobs.label_misinformation.app.pipeline import parse_response


def test_parse_response():
    assert parse_response("Score:10, Reason: score too low") == (10, "score too low")
    assert parse_response("Score: 10, Reason: score too low") == (10, "score too low")
    assert parse_response(
        "Score:10, Reason:The text promotes climate change misinformation by denying the existence of climate change and suggesting that the government is manipulating energy prices and taxes, implying that the high cost of energy is not due to environmental policies but rather government mismanagement. This undermines the scientific consensus on climate change and the need for clean energy initiatives."
    ) == (
        10,
        "The text promotes climate change misinformation by denying the existence of climate change and suggesting that the government is manipulating energy prices and taxes, implying that the high cost of energy is not due to environmental policies but rather government mismanagement. This undermines the scientific consensus on climate change and the need for clean energy initiatives.",
    )
    assert parse_response(
        'Score: 10, Reason: The text promotes climate change misinformation by stating that masks were "possibly dangerous" and that studies from USP (Universidade de São Paulo) confirm this, which undermines the scientific consensus on the effectiveness of masks during the COVID-19 pandemic. This misinformation can lead to public distrust in health measures and scientific research.'
    ) == (
        10,
        'The text promotes climate change misinformation by stating that masks were "possibly dangerous" and that studies from USP (Universidade de São Paulo) confirm this, which undermines the scientific consensus on the effectiveness of masks during the COVID-19 pandemic. This misinformation can lead to public distrust in health measures and scientific research.',
    )
