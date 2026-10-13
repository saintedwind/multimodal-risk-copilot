from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_loads_and_report_is_explicit():
    script = Path(__file__).resolve().parents[1] / "src/app/dashboard.py"
    app = AppTest.from_file(str(script), default_timeout=60).run()
    assert not app.exception
    assert len(app.metric) == 4
    assert any("deterministic" in c.value for c in app.caption)
