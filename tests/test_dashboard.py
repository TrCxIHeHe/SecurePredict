"""Dashboard smoke test (Streamlit AppTest) - no browser needed."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py")


def test_app_loads_without_exception():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception


def test_machine_health_prediction_shows_explanation():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Machine Health").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("SHAP for your input" in s.value for s in at.subheader)
