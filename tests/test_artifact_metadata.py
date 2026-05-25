import json
from pathlib import Path


def test_artifact_metadata_files_exist_and_are_nonempty():
    paths = [
        Path("CITATION.cff"),
        Path(".zenodo.json"),
        Path("ARTIFACT.md"),
        Path("docs/artifact_checklist.md"),
    ]
    for path in paths:
        assert path.exists()
        assert path.read_text().strip()


def test_zenodo_json_is_valid_json():
    metadata = json.loads(Path(".zenodo.json").read_text())
    assert metadata["title"].startswith("ExplainBench")
    assert metadata["upload_type"] == "software"
    assert metadata["license"] == "MIT"


def test_citation_cff_contains_required_fields():
    text = Path("CITATION.cff").read_text()
    for field in [
        "cff-version:",
        "message:",
        "title:",
        "authors:",
        "repository-code:",
        "license:",
    ]:
        assert field in text


def test_readme_states_current_reproducible_scope():
    text = Path("README.md").read_text()
    assert "COMPAS" in text
    assert "implemented and reproducible" in text
    assert "Adult Income" in text
    assert "planned, not yet included" in text
    assert "LendingClub" in text
