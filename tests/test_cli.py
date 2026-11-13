import pytest

from sctag.cli import parser


def test_commands_parse():
    p = parser()
    assert p.parse_args(["run", "someone", "--at", "23:00"]).at == "23:00"
    assert p.parse_args(["fetch", "someone", "--since", "2016-10-27"]).since == "2016-10-27"
    assert p.parse_args(["analyze", "sets", "--sample", "5"]).sample == 5
    assert p.parse_args(["report"]).kind is None
    assert p.parse_args(["serve", "--port", "8766"]).port == 8766
    with pytest.raises(SystemExit):
        p.parse_args(["analyze", "playlists"])
    with pytest.raises(SystemExit):
        p.parse_args(["fetch", "someone", "--since", "27/10/2016"])
