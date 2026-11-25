from pathlib import Path


def test_local_import_preserves_files_and_resolves_companions(client, monkeypatch, tmp_path):
    from backend import data_store
    from backend.proxy.commands import _resolve_llama_model_source, _resolve_mmproj_path, _resolve_draft_companion
    from backend.proxy import client as proxy_client
    from unittest.mock import AsyncMock
    from types import SimpleNamespace

    store = data_store.DataStore(config_dir=str(tmp_path / 'config'))
    monkeypatch.setattr(data_store, '_store', store)
    root = tmp_path / 'models'
    root.mkdir()
    monkeypatch.setenv('STUDIO_LOCAL_MODEL_ROOTS', str(root))
    weight = root / 'model-Q4_K_M.gguf'
    mmproj = root / 'model-mmproj.gguf'
    mtp = root / 'model-mtp.gguf'
    for path in (weight, mmproj, mtp):
        path.write_bytes(b'GGUFtest')
    payload = {'weight_path': str(weight), 'mmproj_path': str(mmproj), 'mtp_path': str(mtp)}
    response = client.post('/api/models/import-local', json=payload)
    assert response.status_code == 200, response.text
    record = store.get_model(response.json()['id'])
    assert _resolve_llama_model_source(record)[0] == str(weight)
    assert _resolve_mmproj_path(record, None, None) == str(mmproj)
    assert _resolve_draft_companion(record, None, None) == (str(mtp), 'draft-mtp')
    assert client.post('/api/models/import-local', json=payload).status_code == 409
    monkeypatch.setattr(proxy_client, 'get_proxy_client', lambda: SimpleNamespace(
        get_running_models=AsyncMock(return_value={'running': []})
    ))
    assert client.delete('/api/models/' + record['id']).status_code == 200
    assert store.get_model(record['id']) is None
    assert all(path.read_bytes() == b'GGUFtest' for path in (weight, mmproj, mtp))
    weight.unlink()
    assert _resolve_llama_model_source(record)[0] is None


def test_local_import_rejects_escape_bad_file_and_shards(client, monkeypatch, tmp_path):
    from backend import data_store
    store = data_store.DataStore(config_dir=str(tmp_path / 'config'))
    monkeypatch.setattr(data_store, '_store', store)
    root = tmp_path / 'models'
    root.mkdir()
    monkeypatch.setenv('STUDIO_LOCAL_MODEL_ROOTS', str(root))
    outside = tmp_path / 'outside.gguf'
    outside.write_bytes(b'GGUFtest')
    link = root / 'escape.gguf'
    link.symlink_to(outside)
    bad = root / 'bad.gguf'
    bad.write_bytes(b'nope')
    shard = root / 'model-00001-of-00002.gguf'
    shard.write_bytes(b'GGUFtest')
    for path in (outside, link, bad, shard):
        assert client.post('/api/models/import-local', json={'weight_path': str(path)}).status_code == 400
    assert not store.list_models()
    assert outside.read_bytes() == b'GGUFtest'
