import pytest
from pydantic import ValidationError

from sec_qa.extract import FilingExtraction


def test_valid_extraction_roundtrips_to_json():
    e = FilingExtraction.model_validate({
        "ticker": "AAPL",
        "revenue": [{"period": "FY2025", "amount_usd_millions": 391035.0, "evidence": "x"}],
    })
    assert FilingExtraction.model_validate_json(e.model_dump_json()) == e


def test_invalid_amount_rejected():
    with pytest.raises(ValidationError):
        FilingExtraction.model_validate(
            {"ticker": "A", "revenue": [{"period": "FY1", "amount_usd_millions": "lots",
                                         "evidence": "x"}]})


def test_too_many_risks_rejected():
    risks = [{"title": str(i), "summary": "s"} for i in range(11)]
    with pytest.raises(ValidationError):
        FilingExtraction.model_validate({"ticker": "A", "risk_factors": risks})
