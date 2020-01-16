import pytest
import altair_transform.driver


@pytest.fixture(scope="session")
def driver():
    try:
        from altair_saver import SeleniumSaver
    except (ImportError, ModuleNotFoundError):
        pytest.skip("altair_saver not importable; cannot run driver tests.")
    if not SeleniumSaver.enabled():
        pytest.skip("selenium not properly configured; cannot run driver tests.")
    yield altair_transform.driver
    SeleniumSaver._stop_serving()
