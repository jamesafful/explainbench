from pathlib import Path


def test_core_documentation_files_exist():
    docs = [
        Path("docs/benchmark_protocol.md"),
        Path("docs/metrics.md"),
        Path("docs/datasets.md"),
        Path("docs/reproducibility.md"),
    ]

    for path in docs:
        assert path.exists()
        assert path.read_text().strip()


def test_reproducibility_doc_mentions_current_outputs():
    text = Path("docs/reproducibility.md").read_text()

    assert "results/compas_multiseed_benchmark.csv" in text
    assert "results/compas_multiseed_summary.csv" in text
    assert "paper/tables" in text
    assert "paper/figures" in text
