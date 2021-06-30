import pytest
import functs as f

# Tests
def test_is_body_first():
    assert f.where_is_body(["mybody"]) == [0]

def test_is_body_second():
    assert f.where_is_body(["nothing", "mybody"]) == [1]

def test_is_body_list():
    assert f.where_is_body(["nothing", "myBody", "anotherbODy"]) == [1,2]


@pytest.mark.parametrize("no_body", [
    [""],
    ["a"],
    ["Never odd or even"],
    ['1', '2']
])

def test_is_not_body(no_body):
    assert f.where_is_body(no_body) == []


def test_is_not_wrongtype():
    with pytest.raises(TypeError):
        f.where_is_body([1]) 
    
def test_is_not_wrongtype():
    with pytest.raises(ValueError):
        f.where_is_body([]) 

@pytest.mark.parametrize("notlist", [
    1,
    "stri",
    "body",
    ('body')
])

def test_is_not_list(notlist):
    with pytest.raises(ValueError):
        f.where_is_body(notlist) 
