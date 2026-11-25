def _local_import(client, monkeypatch, tmp_path, **extra):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("STUDIO_LOCAL_MODEL_ROOTS", str(root))
    weight = root / "model-Q4_K_M.gguf"
    weight.write_bytes(b"GGUFtest")
    response = client.post(
        "/api/models/import-local",
        json={"weight_path": str(weight), **extra},
    )
    return store, response


def test_local_import_stores_optional_huggingface_repo_without_changing_local_identity(
    client, monkeypatch, tmp_path
):
    store, response = _local_import(
        client, monkeypatch, tmp_path,
        huggingface_repo="https://huggingface.co/Owner/Model-GGUF/tree/main",
    )
    assert response.status_code == 200, response.text
    record = store.get_model(response.json()["id"])
    assert record["id"].startswith("local-")
    assert record["source"] == {
        "provider": "local",
        "id": record["id"],
        "huggingface_repo": "Owner/Model-GGUF",
    }
    assert record["local_artifacts"]["weight"].endswith("model-Q4_K_M.gguf")


def test_local_import_enriches_from_huggingface_and_selected_companions(
    client, monkeypatch, tmp_path
):
    from backend import data_store
    from backend.models import hub

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("STUDIO_LOCAL_MODEL_ROOTS", str(root))
    weight = root / "vision-Q4_K_M.gguf"
    mmproj = root / "vision-mmproj-Q8_0.gguf"
    mtp = root / "vision-mtp.gguf"
    for path in (weight, mmproj, mtp):
        path.write_bytes(b"GGUFtest")
    async def details(_repo):
        return {
            "author": "Owner",
            "pipeline_tag": "text-generation",
            "sha": "revision123",
            "tags": ["gguf", "vision"],
            "architecture": "qwen3",
        }

    monkeypatch.setattr(hub, "get_model_details", details)
    response = client.post("/api/models/import-local", json={
        "weight_path": str(weight),
        "mmproj_path": str(mmproj),
        "mtp_path": str(mtp),
        "huggingface_repo": "Owner/Model-GGUF",
    })
    assert response.status_code == 200, response.text
    record = store.get_model(response.json()["id"])
    assert record["source"]["provider"] == "local"
    assert record["huggingface_id"] == "Owner/Model-GGUF"
    assert record["author"] == "Owner"
    assert record["pipeline_tag"] == "text-generation"
    assert record["hf_revision"] == "revision123"
    assert record["model_type"] == "qwen3"
    assert record["mmproj_filename"] == mmproj.name
    assert record["mtp_filename"] == mtp.name
    assert record["capabilities"]["vision"] is True
    assert record["capabilities"]["mtp_draft"] is True


def test_local_source_endpoint_sets_and_clears_only_huggingface_provenance(
    client, monkeypatch, tmp_path
):
    store, response = _local_import(client, monkeypatch, tmp_path)
    assert response.status_code == 200, response.text
    model_id = response.json()["id"]
    source_url = f"/api/models/{model_id}/source"
    updated = client.put(source_url, json={"huggingface_repo": "https://huggingface.co/A/B"})
    assert updated.status_code == 200, updated.text
    assert updated.json()["source"] == {
        "provider": "local", "id": model_id, "huggingface_repo": "A/B"
    }
    cleared = client.put(source_url, json={"huggingface_repo": None})
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["source"] == {"provider": "local", "id": model_id}
    assert store.get_model(model_id)["source"]["provider"] == "local"


def test_local_import_rejects_noncanonical_huggingface_sources(client, monkeypatch, tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("STUDIO_LOCAL_MODEL_ROOTS", str(root))
    weight = root / "model.gguf"
    weight.write_bytes(b"GGUFtest")
    invalid = [
        "http://huggingface.co/a/b",
        "https://example.com/a/b",
        "https://huggingface.co:bad/a/b",
        "https://huggingface.co:443/a/b",
        "https://huggingface.co/a/b/discussions",
        "a/b?download=true",
        "../b/c",
        "a/b/c",
    ]
    for value in invalid:
        response = client.post(
            "/api/models/import-local",
            json={"weight_path": str(weight), "huggingface_repo": value},
        )
        assert response.status_code == 400, (value, response.text)


def test_source_endpoint_rejects_malformed_url_without_mutating_record(
    client, monkeypatch, tmp_path
):
    store, response = _local_import(client, monkeypatch, tmp_path)
    assert response.status_code == 200, response.text
    model_id = response.json()["id"]
    before = store.get_model(model_id)
    invalid = ["https://huggingface.co:bad/a/b", "https://huggingface.co:443/a/b", "https://evil.test/a/b"]
    for value in invalid:
        result = client.put(
            f"/api/models/{model_id}/source",
            json={"huggingface_repo": value},
        )
        assert result.status_code == 400, (value, result.text)
        assert store.get_model(model_id)["source"] == before["source"]
