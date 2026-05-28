import pytest

from kriterion.data import ROCData
from kriterion.fit import fit
from kriterion.models import SignalDetection


def test_evsdt_dunn2011():
    # Dunn (2011)
    data = ROCData(
        signal=[1230, 496, 358, 272, 215, 165],
        noise=[111, 216, 349, 540, 625, 895],
    )

    model = SignalDetection(data)
    result = fit(model)

    assert result.dof == 4
    assert result.log_likelihood == pytest.approx(-8659.73, abs=0.05)
    assert result.g2 == pytest.approx(82.36, abs=0.05)

    assert model.d == pytest.approx(1.37, abs=0.05)
    assert model.n_params == 6
    assert model.dof == 4
