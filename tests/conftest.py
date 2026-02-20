"""Shared fixtures for the test suite."""

import pandas as pd
import pytest


@pytest.fixture()
def sample_dataframe() -> pd.DataFrame:
    """A small synthetic dataframe that mirrors the real dataset schema."""
    return pd.DataFrame(
        {
            "Numtppd": [0, 1, 0, 2, 0, 1, 0, 0, 3, 0],
            "Numtpbi": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            "Indtppd": [0, 500, 0, 1200, 0, 300, 0, 0, 800, 0],
            "Indtpbi": [0, 0, 700, 0, 0, 0, 0, 400, 0, 0],
            "Exposure": [1.0, 0.8, 1.0, 0.5, 1.0, 1.0, 0.3, 1.0, 0.9, 1.0],
            "Power": [50, 60, 70, 80, 50, 60, 70, 80, 50, 60],
            "CalYear": [
                "2010",
                "2010",
                "2011",
                "2011",
                "2010",
                "2011",
                "2010",
                "2011",
                "2010",
                "2011",
            ],
            "Gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
            "Type": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "Category": ["C1", "C2", "C1", "C2", "C1", "C2", "C1", "C2", "C1", "C2"],
            "Occupation": ["O1", "O2", "O1", "O2", "O1", "O2", "O1", "O2", "O1", "O2"],
            "SubGroup2": ["S1", "S2", "S1", "S2", "S1", "S2", "S1", "S2", "S1", "S2"],
            "Group2": ["G1", "G2", "G1", "G2", "G1", "G2", "G1", "G2", "G1", "G2"],
            "Group1": ["GG1", "GG2", "GG1", "GG2", "GG1", "GG2", "GG1", "GG2", "GG1", "GG2"],
        }
    )
