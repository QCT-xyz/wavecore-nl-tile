from wavecore_nl.config import settings


def test_defaults_load():
    assert settings.backend in {"sim", "hardware"}
    assert settings.modes >= 1
