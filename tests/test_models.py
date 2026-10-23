"""Input validation: whitespace and control characters, emptiness and length."""

import pytest
from pydantic import ValidationError

from app.models import AskRequest, SummarizeRequest


def test_whitespace_and_control_characters_are_cleaned():
    assert AskRequest(question="  What\x00 is\n\tDNS?  ").question == "What is DNS?"
    assert SummarizeRequest(topic="\x07DNS\x1b  caching").topic == "DNS caching"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", "\x00\x01"])
def test_empty_after_cleaning_is_rejected(bad: str):
    with pytest.raises(ValidationError):
        AskRequest(question=bad)
    with pytest.raises(ValidationError):
        SummarizeRequest(topic=bad)


def test_length_limit_is_enforced_after_cleaning():
    assert AskRequest(question="x" * 500).question == "x" * 500
    with pytest.raises(ValidationError, match="at most 500"):
        AskRequest(question="x" * 501)
    # Control characters do not count toward the limit once stripped.
    assert len(AskRequest(question="x" * 500 + "\x00\x00").question) == 500
