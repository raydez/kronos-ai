import kronos_ai


def test_package_imports() -> None:
    assert kronos_ai.__version__.startswith("2.")


def test_phase1_subpackages_exist() -> None:
    import importlib

    for mod in (
        "kronos_ai.domain",
        "kronos_ai.data",
        "kronos_ai.forecast",
        "kronos_ai.forecast.backends.kronos",
        "kronos_ai.infrastructure.providers",
        "kronos_ai.infrastructure.persistence",
        "kronos_ai.cli",
    ):
        importlib.import_module(mod)
