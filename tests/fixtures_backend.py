import pytest
from VolumeRaytraceLFM.simulations import BackEnds


BACKENDS = {
    'numpy': BackEnds.NUMPY,
    'pytorch': BackEnds.PYTORCH
}


@pytest.fixture(scope="module")
def backend_fixture(request):
    try:
        return BACKENDS[request.param]
    except KeyError:
        raise ValueError(f"Invalid backend: {request.param}")
