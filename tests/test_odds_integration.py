"""
Basic tests for odds integration components
"""
import os
from datetime import datetime
from src.data.odds_schema import OddsRecord, OddsData
from src.utils.odds_converter import odds_to_probability, remove_margin, calculate_margin

def test_odds_record_validation():
    """Test OddsRecord validation"""
    # Valid odds
    record = OddsRecord(
        date=datetime.now(),
        home_team="AIK",
        away_team="Djurgarden",
        home_odds=2.0,
        draw_odds=3.0,
        away_odds=3.5,
        bookmaker="test"
    )
    assert record.home_odds == 2.0
    assert record.draw_odds == 3.0
    assert record.away_odds == 3.5

def test_odds_to_probability():
    """Test odds to probability conversion"""
    assert abs(odds_to_probability(2.0) - 0.5) < 0.001
    assert abs(odds_to_probability(4.0) - 0.25) < 0.001
    
def test_remove_margin():
    """Test margin removal"""
    probs = remove_margin(2.0, 3.5, 4.0)
    assert abs(sum(probs) - 1.0) < 0.001
    assert all(0 < p < 1 for p in probs)

def test_calculate_margin():
    """Test margin calculation"""
    margin = calculate_margin(2.0, 3.0, 3.5)
    assert margin > 0  # Should be positive
    assert margin < 20  # Should be reasonable

def test_odds_data_storage():
    """Test OddsData storage and retrieval"""
    odds_data = OddsData()
    
    record = OddsRecord(
        date=datetime.now(),
        home_team="AIK",
        away_team="Djurgarden", 
        home_odds=2.0,
        draw_odds=3.0,
        away_odds=3.5,
        bookmaker="test"
    )
    
    odds_data.add_match_odds(record)
    retrieved = odds_data.get_match_odds("AIK", "Djurgarden", datetime.now())
    assert retrieved is not None
    assert retrieved.home_odds == 2.0

if __name__ == "__main__":
    # Run basic tests
    test_odds_record_validation()
    test_odds_to_probability()
    test_remove_margin()
    test_calculate_margin()
    test_odds_data_storage()
    print("✅ All basic tests passed!")