// Offline harness for dsh-qwen-image-generate: stubs fetch + ctx services, exercises both tools.
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==',
  'base64',
)
// A minimal MP4 header: box size + 'ftyp' + brand bytes.
const MP4 = Buffer.concat([
  Buffer.from([0x00, 0x00, 0x00, 0x18]),
  Buffer.from('ftypisom'),
  Buffer.from([0x00, 0x00, 0x02, 0x00]),
  Buffer.from('isomiso2avc1mp41'),
])
// The attachment service normalizes what it stores: the reference's byte size
// and media type can differ from the downloaded source bytes.
const STORED = { bytes: 12, mediaType: 'image/webp', width: 1024, height: 1024 }
const IMAGE_BODY = {
  output: {
    rewrite_status: 'not_use',
    choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: [{ image: 'https://cdn.example/a.png' }] } }],
  },
  usage: { output_width: 1024, output_height: 1024, output_image_count: 1 },
  request_id: 'req-1',
}

const calls = []
let taskOnePolls = 0
let rejectBatchField = false
let singleImageSerial = 0
let transientFailuresLeft = 0
const json = (value) => new Response(JSON.stringify(value), { status: 200, headers: { 'content-type': 'application/json' } })
const failure = (status, value) => new Response(JSON.stringify(value), { status, headers: { 'content-type': 'application/json' } })

globalThis.fetch = async (url, init) => {
  const target = String(url)
  calls.push({ url: target, init })
  if (target.endsWith('/video-synthesis')) {
    const body = JSON.parse(String(init?.body))
    const taskId = body.input.prompt === 'hang forever' ? 'task-pending' : 'task-1'
    return json({ request_id: 'req-v', output: { task_id: taskId, task_status: 'PENDING' } })
  }
  if (target.includes('/tasks/task-1')) {
    taskOnePolls += 1
    if (taskOnePolls < 2) return json({ output: { task_id: 'task-1', task_status: 'RUNNING' } })
    return json({
      output: { task_id: 'task-1', task_status: 'SUCCEEDED', video_url: 'https://cdn.example/v.mp4' },
      usage: { video_duration: 5, video_ratio: '16:9' },
    })
  }
  if (target.includes('/tasks/task-pending')) return json({ output: { task_id: 'task-pending', task_status: 'RUNNING' } })
  if (target === 'https://cdn.example/v.mp4') {
    return new Response(MP4, { status: 200, headers: { 'content-type': 'video/mp4' } })
  }
  if (target.startsWith('https://token-plan')) {
    const body = JSON.parse(String(init?.body))
    const prompt = body.input?.messages?.[0]?.content?.find((part) => typeof part.text === 'string')?.text ?? ''
    if (transientFailuresLeft > 0) {
      transientFailuresLeft -= 1
      return failure(503, { code: 'ServiceUnavailable', message: 'service is busy' })
    }
    if (prompt === 'unauthorized') {
      return failure(401, { code: 'InvalidApiKey', message: 'invalid api key', request_id: 'req-401' })
    }
    if (rejectBatchField && body.parameters?.n !== undefined) {
      return failure(400, { code: 'InvalidParameter', message: 'n is not supported by this model', request_id: 'req-400' })
    }
    if (rejectBatchField) {
      singleImageSerial += 1
      return json({
        ...IMAGE_BODY,
        output: {
          ...IMAGE_BODY.output,
          choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: [{ image: 'https://cdn.example/single-' + singleImageSerial + '.png' }] } }],
        },
      })
    }
    return json(IMAGE_BODY)
  }
  return new Response(PNG, { status: 200, headers: { 'content-type': 'image/png' } })
}

// Approximate DSH's enforced JSON Schema subset (dsh-tools/src/json-schema.ts):
// constraint keywords type/oneOf/properties/required/additionalProperties/items/enum/const,
// annotations description/title/default/examples.
const CONSTRAINT_KEYWORDS = new Set(['type', 'oneOf', 'properties', 'required', 'additionalProperties', 'items', 'enum', 'const'])
const ANNOTATION_KEYWORDS = new Set(['description', 'title', 'default', 'examples'])
const schemaViolations = []
const checkSchema = (node, path) => {
  if (node === null || typeof node !== 'object' || Array.isArray(node)) {
    schemaViolations.push(path + ' is not an object')
    return
  }
  for (const key of Object.keys(node)) {
    if (!CONSTRAINT_KEYWORDS.has(key) && !ANNOTATION_KEYWORDS.has(key)) {
      schemaViolations.push(path + '.' + key + ' is not a supported keyword')
    }
  }
  if (node.required !== undefined) {
    if (!Array.isArray(node.required) || node.required.some((entry) => typeof entry !== 'string')) {
      schemaViolations.push(path + '.required must be an array of strings')
    } else {
      for (const key of node.required) {
        if (node.properties === undefined || !Object.hasOwn(node.properties, key)) {
          schemaViolations.push(path + '.required names ' + key + ' which is not in properties')
        }
      }
    }
  }
  if (node.additionalProperties !== undefined && typeof node.additionalProperties !== 'boolean') {
    schemaViolations.push(path + '.additionalProperties must be a boolean')
  }
  if (node.properties !== undefined) {
    if (node.type !== 'object') schemaViolations.push(path + '.properties requires type object')
    for (const key of Object.keys(node.properties)) checkSchema(node.properties[key], path + '.' + key)
  }
  if (node.items !== undefined) {
    if (node.type !== 'array') schemaViolations.push(path + '.items requires type array')
    checkSchema(node.items, path + '.items')
  }
  if (node.enum !== undefined && !Array.isArray(node.enum)) schemaViolations.push(path + '.enum must be an array')
}

const dir = await mkdtemp(join(tmpdir(), 'qwen-media-test-'))
const saved = []
const failures = []
const expect = (label, actual, expected) => {
  const ok = JSON.stringify(actual) === JSON.stringify(expected)
  if (!ok) failures.push(label + ': expected ' + JSON.stringify(expected) + ', got ' + JSON.stringify(actual))
  console.log((ok ? 'PASS ' : 'FAIL ') + label)
}
const mod = await import('./index.js')
const registered = {}
const ctx = {
  effect: (fn) => fn(),
  tools: { register: (definition) => { registered[definition.name] = definition; return () => {} } },
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
mod.apply(ctx, { videoPollIntervalMs: 50, videoWaitMs: 1000 })
const exec = { agent: { id: 'a', options: {}, session: { requestHeader: () => ({ config: { provider: 'p', model: 'm' } }) } }, signal: undefined }
for (const definition of Object.values(registered)) {
  checkSchema(definition.parameters, definition.name + '.parameters')
  checkSchema(definition.output.schema, definition.name + '.output.schema')
}
expect('declared schemas stay inside the enforced subset', schemaViolations, [])

// --- image tool -----------------------------------------------------------
const imageValue = await registered.qwen_image_generate.execute({ prompt: 'a red bicycle' }, exec)
const imageBlocks = registered.qwen_image_generate.output.render({ prompt: 'x' }, imageValue)
const imageRequest = calls.find((call) => call.url.includes('/multimodal-generation/generation'))
expect('image default model is wan2.7-image-pro', JSON.parse(imageRequest.init.body).model, 'wan2.7-image-pro')
expect('saved file size', (await readFile(imageValue.images[0].path)).byteLength, PNG.byteLength)
expect('result bytes is the saved file size', imageValue.images[0].bytes, PNG.byteLength)
expect('result attachmentBytes is the normalized size', imageValue.images[0].attachmentBytes, STORED.bytes)
expect('result mediaType is the normalized type', imageValue.images[0].mediaType, STORED.mediaType)
expect('render block count', imageBlocks.length, 2)
expect('render image block uses normalized bytes', imageBlocks[1].attachment.bytes, STORED.bytes)
expect('render image block uses normalized mediaType', imageBlocks[1].attachment.mediaType, STORED.mediaType)

// --- video tool: submit, poll, download -----------------------------------
const videoValue = await registered.qwen_video_generate.execute({ prompt: 'a cat on a roof', wait_seconds: 5 }, exec)
const videoBlocks = registered.qwen_video_generate.output.render({ prompt: 'x' }, videoValue)
const videoSubmit = calls.find((call) => call.url.endsWith('/video-synthesis'))
const submitBody = JSON.parse(videoSubmit.init.body)
expect('video default model', submitBody.model, 'wan3.0-video')
expect('video async header', videoSubmit.init.headers['X-DashScope-Async'], 'enable')
expect('video auth header', videoSubmit.init.headers.Authorization, 'Bearer sk-sp-test')
expect('video parameters', submitBody.parameters, { resolution: '720P', duration: 5, ratio: '16:9' })
expect('video input', submitBody.input, { prompt: 'a cat on a roof' })
expect('video task queried twice before success', taskOnePolls, 2)
expect('video status', videoValue.status, 'succeeded')
expect('video task id', videoValue.taskId, 'task-1')
expect('video usage duration', videoValue.video.durationSeconds, 5)
expect('video usage ratio', videoValue.video.ratio, '16:9')
expect('video file size', (await readFile(videoValue.video.path)).byteLength, MP4.byteLength)
expect('video media type', videoValue.video.mediaType, 'video/mp4')
expect('video render mentions path', videoBlocks[0].text.includes(videoValue.video.path), true)
expect('video presentation meta', registered.qwen_video_generate.output.presentationMeta({}, videoValue), { paths: [videoValue.video.path] })

// --- video tool: still pending returns the task id ------------------------
const pendingValue = await registered.qwen_video_generate.execute({ prompt: 'hang forever', wait_seconds: 1 }, exec)
const pendingBlocks = registered.qwen_video_generate.output.render({ prompt: 'x' }, pendingValue)
expect('pending status', pendingValue.status, 'pending')
expect('pending task id', pendingValue.taskId, 'task-pending')
expect('pending has no video', pendingValue.video, undefined)
expect('pending render mentions task id', pendingBlocks[0].text.includes('task-pending'), true)

// --- video tool: first-frame media keeps the frame shape ------------------
calls.length = 0
taskOnePolls = 0
await registered.qwen_video_generate.execute({
  prompt: 'the cat turns its head',
  media: [{ type: 'first_frame', url: 'https://cdn.example/a.png' }],
  wait_seconds: 5,
}, exec)
const frameSubmit = JSON.parse(calls.find((call) => call.url.endsWith('/video-synthesis')).init.body)
expect('first-frame media forwarded', frameSubmit.input.media, [{ type: 'first_frame', url: 'https://cdn.example/a.png' }])
expect('first-frame ratio falls back to adaptive', frameSubmit.parameters.ratio, 'adaptive')

// --- video tool: HappyHorse drops wan-only parameters ---------------------
calls.length = 0
taskOnePolls = 0
await registered.qwen_video_generate.execute({
  prompt: 'the cat turns its head',
  model: 'happyhorse-1.1-i2v',
  audio: true,
  media: [{ type: 'first_frame', url: 'https://cdn.example/a.png' }],
  wait_seconds: 5,
}, exec)
const horseParams = JSON.parse(calls.find((call) => call.url.endsWith('/video-synthesis')).init.body).parameters
expect('happyhorse omits ratio', horseParams.ratio, undefined)
expect('happyhorse omits audio', horseParams.audio, undefined)
expect('happyhorse keeps resolution and duration', [horseParams.resolution, horseParams.duration], ['720P', 5])

// --- video tool: file media forces prompt rewriting -----------------------
calls.length = 0
taskOnePolls = 0
await registered.qwen_video_generate.execute({
  prompt: 'make a product ad from this deck',
  media: [{ type: 'file', url: 'https://cdn.example/deck.pptx' }],
  wait_seconds: 5,
}, exec)
const fileParams = JSON.parse(calls.find((call) => call.url.endsWith('/video-synthesis')).init.body).parameters
expect('file media forces prompt_extend', fileParams.prompt_extend, true)

// --- video tool: task_id resumes without a new submission -----------------
calls.length = 0
taskOnePolls = 1
const resumed = await registered.qwen_video_generate.execute({ task_id: 'task-1', wait_seconds: 5 }, exec)
expect('resume skips submission', calls.some((call) => call.url.endsWith('/video-synthesis')), false)
expect('resume downloads the finished video', resumed.status, 'succeeded')
expect('resume keeps the task id', resumed.taskId, 'task-1')

// --- argument validation --------------------------------------------------
let rejected = undefined
try {
  await registered.qwen_video_generate.execute({ wait_seconds: 1 }, exec)
} catch (error) {
  rejected = error?.code
}
expect('empty request rejected', rejected, 'INVALID_ARGS')

// --- output_dir stays inside the workspace --------------------------------
let escaped = undefined
try {
  await registered.qwen_image_generate.execute({ prompt: 'a red bicycle', output_dir: '../escape' }, exec)
} catch (error) {
  escaped = error?.code
}
expect('output_dir outside the workspace rejected', escaped, 'INVALID_ARGS')

// --- provider request id reaches the failure message ----------------------
let unauthorized = ''
try {
  await registered.qwen_image_generate.execute({ prompt: 'unauthorized' }, exec)
} catch (error) {
  unauthorized = error?.message ?? ''
}
expect('failure carries the provider request id', unauthorized.includes('req-401'), true)

// --- concurrent images never share a file name ----------------------------
const [firstImage, secondImage] = await Promise.all([
  registered.qwen_image_generate.execute({ prompt: 'a blue bus' }, exec),
  registered.qwen_image_generate.execute({ prompt: 'a blue bus' }, exec),
])
expect('concurrent images get distinct files', firstImage.images[0].path !== secondImage.images[0].path, true)

// --- count above 1 falls back to single requests --------------------------
calls.length = 0
rejectBatchField = true
singleImageSerial = 0
const batchValue = await registered.qwen_image_generate.execute({ prompt: 'three lanterns', count: 3 }, exec)
const batchParams = calls
  .filter((call) => call.url.includes('/multimodal-generation'))
  .map((call) => JSON.parse(String(call.init.body)).parameters)
expect('batch fallback counts the rejected attempt', batchParams.length, 4)
expect('batch fallback drops the n field', batchParams.slice(1).every((parameters) => parameters.n === undefined), true)
expect('batch fallback keeps every image', batchValue.images.length, 3)
rejectBatchField = false

// --- pending tasks are ledgered, finished ones removed --------------------
const ledger = JSON.parse(await readFile(join(dir, 'qwen-videos', '.pending-tasks.json'), 'utf8'))
expect('pending task is written to the ledger', Object.hasOwn(ledger, 'task-pending'), true)
expect('finished task leaves the ledger', Object.hasOwn(ledger, 'task-1'), false)

// --- image tool: several prompts in one call ------------------------------
calls.length = 0
const seriesValue = await registered.qwen_image_generate.execute({
  prompts: ['a rat holding coins', 'an ox in shallow water'],
  name_prefix: ['01_rat', '02_ox'],
}, exec)
const seriesCalls = calls.filter((call) => String(call.url).includes('/multimodal-generation'))
expect('one request per prompt', seriesCalls.length, 2)
expect('series returns both images', seriesValue.images.length, 2)
expect('series names follow the prefix array', [
  seriesValue.images[0].path.endsWith('01_rat.png'),
  seriesValue.images[1].path.endsWith('02_ox.png'),
], [true, true])
expect('series reports the prompt index', seriesValue.images.map((image) => image.promptIndex), [0, 1])

// --- image tool: a single name prefix numbers the files -------------------
calls.length = 0
rejectBatchField = true
singleImageSerial = 0
const prefixed = await registered.qwen_image_generate.execute({ prompt: 'a white horse', count: 2, name_prefix: 'horse' }, exec)
expect('single prefix numbers the files', [
  prefixed.images[0].path.endsWith('horse_01.png'),
  prefixed.images[1].path.endsWith('horse_02.png'),
], [true, true])
rejectBatchField = false

// --- image tool: manifest ledger -----------------------------------------
const manifestLines = (await readFile(join(dir, 'qwen-images', '.manifest.jsonl'), 'utf8'))
  .trim().split(String.fromCharCode(10)).map((line) => JSON.parse(line))
const manifestHit = manifestLines.find((line) => line.file === seriesValue.images[0].path)
expect('manifest records the prompt', manifestHit.prompt, 'a rat holding coins')
expect('manifest records the model', manifestHit.model, 'wan2.7-image-pro')
expect('manifest path is returned', seriesValue.manifestPath, join(dir, 'qwen-images', '.manifest.jsonl'))

// --- image tool: negative_prompt on a model that ignores it ---------------
calls.length = 0
const ignoredNegative = await registered.qwen_image_generate.execute({ prompt: 'a grey cat', negative_prompt: 'blur' }, exec)
expect('negative_prompt is reported as ignored', Array.isArray(ignoredNegative.warnings) && ignoredNegative.warnings.some((warning) => warning.includes('negative_prompt was not sent')), true)
const ignoredRequest = JSON.parse(String(calls.filter((call) => String(call.url).includes('/multimodal-generation')).pop().init.body))
expect('negative_prompt is not sent to wan', ignoredRequest.parameters.negative_prompt, undefined)
const ignoredBlocks = registered.qwen_image_generate.output.render({}, ignoredNegative)
expect('warnings reach the rendered text', ignoredBlocks.some((block) => String(block.text).includes('negative_prompt was not sent')), true)

// --- image tool: batch budget and argument validation ---------------------
const budgetRejected = async (args) => {
  try {
    await registered.qwen_image_generate.execute(args, exec)
    return undefined
  } catch (error) {
    return error?.code
  }
}
expect('prompts times count stays within one call', await budgetRejected({ prompts: ['a', 'b', 'c', 'd'], count: 2 }), 'INVALID_ARGS')
expect('prompt and prompts are exclusive', await budgetRejected({ prompt: 'a cat', prompts: ['a dog'] }), 'INVALID_ARGS')
expect('prefix array must cover every image', await budgetRejected({ prompts: ['a cat', 'a dog'], name_prefix: ['only-one'] }), 'INVALID_ARGS')

// --- image tool: retryable failures are retried ---------------------------
calls.length = 0
transientFailuresLeft = 1
const retried = await registered.qwen_image_generate.execute({ prompt: 'a green frog' }, exec)
expect('transient failure is retried', calls.filter((call) => String(call.url).includes('/multimodal-generation')).length, 2)
expect('retry still returns the image', retried.images.length, 1)
transientFailuresLeft = 0

await rm(dir, { recursive: true, force: true })
if (failures.length > 0) {
  console.error('FAILURES: ' + failures.join(' | '))
  process.exitCode = 1
}
