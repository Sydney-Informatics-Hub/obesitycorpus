import pytest
import functs as f

# Tests
def test_is_month_full():
    assert f.convert_month("January") == "Jan"

def test_is_month_short():
    assert f.convert_month("Jan") == "Jan"

@pytest.mark.parametrize("non_month", [
    "",
    "a",
    1,
    "Never odd or even",
    ['1', '2'],
])

def test_is_not_month_name(non_month):
    assert f.convert_month(non_month) == print("The following month name is not valid: ", non_month)
