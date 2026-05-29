import pytest

from kriterion.data import ROCData
from kriterion.fit import fit
from kriterion.models import SignalDetection, UnequalSignalDetection


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


def test_uvsdt_koen():
    # From Koen's ROC Toolbox, Example 1
    data = ROCData(
        signal=[338, 100, 117, 103, 89, 75, 67, 46, 28, 37],
        noise=[54, 58, 103, 118, 153, 154, 121, 104, 63, 72],
    )

    model = UnequalSignalDetection(data)
    result = fit(model)

    assert result.dof == 7
    assert result.log_likelihood == pytest.approx(-4276.8, abs=0.5)
    assert result.g2 == pytest.approx(2.8042, abs=0.05)

    assert result.aic == pytest.approx(8575.6, abs=1.0)
    assert result.bic == pytest.approx(8637.2, abs=1.0)

    assert model.d == pytest.approx(0.9958, abs=0.01)
    assert model.signal_sd == pytest.approx(1.4055, abs=0.01)
    assert model.n_params == 11
    assert model.dof == 7
