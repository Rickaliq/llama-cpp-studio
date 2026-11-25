import { describe, expect, it } from 'vitest'
import { huggingFaceSourceId, huggingFaceSourceUrl, normalizeHuggingFaceRepo } from './modelSource'

describe('Hugging Face source metadata', () => {
  it('uses an explicit backend source repository before legacy source mappings', () => {
    const model = {
      id: 'local-e86183abe55f70c7c6ff',
      source: { provider: 'local', huggingface_repo: 'https://huggingface.co/team/new-model' },
    }
    expect(huggingFaceSourceId(model)).toBe('team/new-model')
    expect(huggingFaceSourceUrl(model)).toBe('https://huggingface.co/team/new-model')
  })

  it('does not infer a repository from a local identifier and accepts repository URLs', () => {
    expect(huggingFaceSourceId({ id: 'local-e86183abe55f70c7c6ff' })).toBe('')
    expect(normalizeHuggingFaceRepo('https://huggingface.co/team/model/')).toBe('team/model')
    expect(normalizeHuggingFaceRepo('team/model')).toBe('team/model')
    expect(normalizeHuggingFaceRepo('https://example.com/team/model')).toBe('')
    expect(
      huggingFaceSourceUrl({
        id: 'local-e86183abe55f70c7c6ff',
        source: { provider: 'local', huggingface_repo: null },
      }),
    ).toBe('')
  })
})
