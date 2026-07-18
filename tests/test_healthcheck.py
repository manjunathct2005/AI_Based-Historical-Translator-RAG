from health.healthcheck import check_disk_space, run_all_checks


def test_check_disk_space_returns_result():
    result = check_disk_space(".", min_free_gb=0.001)
    assert result.name == "disk_space"
    assert result.ok is True


def test_run_all_checks_returns_list_of_results():
    results = run_all_checks()
    names = [r.name for r in results]
    assert "internet" in names
    assert "gpu" in names
    assert "disk_space" in names
    assert any(n.startswith("import:") for n in names)
