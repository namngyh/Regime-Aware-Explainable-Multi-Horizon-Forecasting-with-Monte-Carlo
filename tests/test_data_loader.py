from pathlib import Path

from raemf_mc.data.loader import load_price_data


def test_parser_handles_split_thousands(tmp_path: Path):
    p = tmp_path / "d.csv"
    p.write_text("Date,Open,High,Low,Close,Volume,,,,\n1/1/2020,1,2,1,1.5,4,200,,,\n", encoding="utf-8")
    df, meta = load_price_data(p)
    assert df.loc[0, "volume"] == 4200
    assert meta["rows_loaded"] == 1
