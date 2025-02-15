import pytest
from VolumeRaytraceLFM.abstract_classes import BackEnds


BACKENDS = {"numpy": BackEnds.NUMPY, "torch": BackEnds.PYTORCH}


@pytest.fixture(scope="module")
def backend_fixture(request):
    try:
        return BACKENDS[request.param]
    except KeyError:
        raise ValueError(f"Invalid backend: {request.param}")
