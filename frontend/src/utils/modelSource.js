export function huggingFaceSourceId(model) {
  if (!model || typeof model !== 'object') return ''

  const sourceRepo = String(model.source?.huggingface_repo || '').trim()
  if (sourceRepo) return normalizeHuggingFaceRepo(sourceRepo)

  const sourceId = String(model.huggingface_id || '').trim()
  return sourceId.includes('/') ? sourceId : ''
}

export function normalizeHuggingFaceRepo(value) {
  const input = String(value || '').trim()
  const match = input.match(/^(?:https?:\/\/)?(?:www\.)?huggingface\.co\/([^?#]+)\/?$/i)
  const repo = (match ? match[1] : input).replace(/\/$/, '')
  return /^[^/]+\/[^/]+$/.test(repo) ? repo : ''
}

export function huggingFaceSourceUrl(model) {
  const sourceId = huggingFaceSourceId(model)
  return sourceId ? `https://huggingface.co/${sourceId}` : ''
}
