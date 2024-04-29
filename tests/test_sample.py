import pytest
from pathlib import Path
import sys

from src.sample import Sample

def test_sample_creating():
    """Test the loading of a Sample instance from a JSON file."""
    sample_path = Path(__file__).parent / "test_config.json"  # Correct path to the JSON config file
    sample = Sample(str(sample_path))
    assert sample is not None
    assert sample.info.ID == "20220426_RM0008_130hpf_fP1_f3"  # Checking if the ID matches
    assert hasattr(sample, 'exp'), "Sample should have 'exp' attribute"

# Running the test if this script is executed directly
if __name__ == "__main__":
    pytest.main()


