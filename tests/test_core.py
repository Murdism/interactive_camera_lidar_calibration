from recalib.core import ping

def test_ping():
    assert ping() == "recalib OK"
