// Offline harness for dsh-qwen-image: stubs fetch + ctx services, exercises execute.
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==',
  'base64',
)
// The attachment service normalizes what it stores: the reference's byte size
// and media type can differ from the downloaded source bytes.
const STORED = { bytes: 12, mediaType: 'image/webp', width: 1024, height: 1024 }
const calls = []
globalThis.fetch = async (url, init) => {
  calls.push({ url: String(url), init })
  if (String(url).startsWith('https://token-plan')) {
    return new Response(JSON.stringify({
      output: {
        rewrite_status: 'not_use',
        choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: [{ image: 'https://cdn.example/a.png' }] } }],
      },
      usage: { output_width: 1024, output_height: 1024, output_image_count: 1 },
      request_id: 'req-1',
    }), { status: 200, headers: { 'content-type': 'application/json' } })
  }
  return new Response(PNG, { status: 200, headers: { 'content-type': 'image/png' } })
}

const dir = await mkdtemp(join(tmpdir(), 'qwen-image-test-'))
const saved = []
const failures = []
const expect = (label, actual, expected) => {
  const ok = JSON.stringify(actual) === JSON.stringify(expected)
  if (!ok) failures.push(label + ': expected ' + JSON.stringify(expected) + ', got ' + JSON.stringify(actual))
  console.log((ok ? 'PASS ' : 'FAIL ') + label)
}
const mod = await import('./index.js')
const ctx = {
  effect: (fn) => fn(),
  tools: { register: (definition) => { ctx.definition = definition; return () => {} } },
  get: (name) => {
    if (name === 'credentials') return { resolve: async () => ({ value: 'sk-sp-test', source: 'store' }) }
    if (name === 'workingDirectory') return { ensure: async () => dir }
    if (name === 'attachments') {
      return {
        saveImage: async ({ data, name: fileName }) => {
          saved.push({ bytes: data.byteLength, fileName })
          return { attachmentId: 'sha256:abc123', ...STORED, name: fileName }
        },
      }
    }
    if (name === 'llm') return { resolveModelInfo: async () => ({ inputModalities: ['text', 'image'] }) }
    return undefined
  },
}
mod.apply(ctx, {})
const exec = { agent: { id: 'a', options: {}, session: { requestHeader: () => ({ config: { provider: 'p', model: 'm' } }) } }, signal: undefined }
const value = await ctx.definition.execute({ prompt: 'a red bicycle' }, exec)
const blocks = ctx.definition.output.render({ prompt: 'x' }, value)
expect('saved file size', (await readFile(value.images[0].path)).byteLength, PNG.byteLength)
expect('result bytes is the saved file size', value.images[0].bytes, PNG.byteLength)
expect('result attachmentBytes is the normalized size', value.images[0].attachmentBytes, STORED.bytes)
expect('result mediaType is the normalized type', value.images[0].mediaType, STORED.mediaType)
expect('render block count', blocks.length, 2)
expect('render image block uses normalized bytes', blocks[1].attachment.bytes, STORED.bytes)
expect('render image block uses normalized mediaType', blocks[1].attachment.mediaType, STORED.mediaType)
console.log('request body:', calls[0].init.body)
console.log('auth:', calls[0].init.headers.Authorization)
await rm(dir, { recursive: true, force: true })
if (failures.length > 0) {
  console.error('FAILURES: ' + failures.join(' | '))
  process.exitCode = 1
}
