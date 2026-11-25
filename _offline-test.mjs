// Offline harness for dsh-qwen-image-generate: stubs fetch + ctx services, exercises both tools.
import { mkdir, mkdtemp, readdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
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
let unknownPolls = 0
let flakyPolls = 0
const json = (value) => new Response(JSON.stringify(value), { status: 200, headers: { 'content-type': 'application/json' } })
const failure = (status, value) => new Response(JSON.stringify(value), { status, headers: { 'content-type': 'application/json' } })
const imageBodyWith = (image) => ({
  ...IMAGE_BODY,
  output: {
    ...IMAGE_BODY.output,
    choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: [{ image }] } }],
  },
})
const videoSucceeded = (taskId) => json({
  output: { task_id: taskId, task_status: 'SUCCEEDED', video_url: 'https://cdn.example/v.mp4' },
  usage: { video_duration: 5, video_ratio: '16:9' },
})

globalThis.fetch = async (url, init) => {
  const target = String(url)
  calls.push({ url: target, init })
  // A hung transfer must reject when its caller cancels, like fetch does. The
  // ref'd timer keeps this process alive: AbortSignal.timeout timers are unref'd,
  // so a bare await would let Node exit before the timeout can fire.
  const hangUntilAborted = (name = 'AbortError') => new Promise((_resolve, reject) => {
    const watchdog = setTimeout(() => reject(new Error('the test transfer was never aborted')), 30_000)
    const abort = () => {
      clearTimeout(watchdog)
      reject(Object.assign(new Error('The operation was aborted'), { name }))
    }
    if (init?.signal?.aborted) abort()
    else init?.signal?.addEventListener('abort', abort, { once: true })
  })
  if (target.endsWith('/video-synthesis')) {
    const body = JSON.parse(String(init?.body))
    const prompt = body.input?.prompt ?? ''
    if (prompt === 'slow submit') return hangUntilAborted('TimeoutError')
    const taskId = prompt === 'hang forever' ? 'task-pending'
      : prompt === 'hang one' ? 'task-pending-1'
        : prompt === 'hang two' ? 'task-pending-2'
          : prompt === 'unknown then ok' ? 'task-unknown'
            : prompt === 'always unknown' ? 'task-dead'
              : prompt === 'flaky query' ? 'task-flaky'
                : prompt === 'slow download' ? 'task-download-slow'
                  : prompt === 'huge video' ? 'task-huge-video'
                    : prompt === 'timeout query' ? 'task-timeout-query'
                      : 'task-1'
    return json({ request_id: 'req-v', output: { task_id: taskId, task_status: 'PENDING' } })
  }
  if (target.includes('/tasks/task-unknown')) {
    unknownPolls += 1
    if (unknownPolls <= 2) return json({ output: { task_id: 'task-unknown' } })
    return videoSucceeded('task-unknown')
  }
  if (target.includes('/tasks/task-dead')) return json({ output: { task_id: 'task-dead' } })
  if (target.includes('/tasks/task-flaky')) {
    flakyPolls += 1
    if (flakyPolls <= 2) return failure(503, { code: 'ServiceUnavailable', message: 'try again' })
    return videoSucceeded('task-flaky')
  }
  if (target.includes('/tasks/task-download-slow')) {
    return json({ output: { task_id: 'task-download-slow', task_status: 'SUCCEEDED', video_url: 'https://cdn.example/slow.mp4' } })
  }
  if (target.includes('/tasks/task-huge-video')) {
    return json({ output: { task_id: 'task-huge-video', task_status: 'SUCCEEDED', video_url: 'https://cdn.example/huge.mp4' } })
  }
  if (target.includes('/tasks/task-timeout-query')) return hangUntilAborted('TimeoutError')
  if (target.includes('/tasks/task-pending-')) {
    return json({ output: { task_id: target.slice(target.lastIndexOf('/') + 1), task_status: 'RUNNING' } })
  }
  if (target.includes('/tasks/task-1')) {
    taskOnePolls += 1
    if (taskOnePolls < 2) return json({ output: { task_id: 'task-1', task_status: 'RUNNING' } })
    return videoSucceeded('task-1')
  }
  if (target.includes('/tasks/task-pending')) return json({ output: { task_id: 'task-pending', task_status: 'RUNNING' } })
  if (target === 'https://cdn.example/v.mp4') {
    return new Response(MP4, { status: 200, headers: { 'content-type': 'video/mp4' } })
  }
  if (target === 'https://cdn.example/slow.mp4') return hangUntilAborted()
  if (target === 'https://cdn.example/huge.mp4') {
    return new Response(MP4, {
      status: 200,
      headers: { 'content-type': 'video/mp4', 'content-length': String(2 * 1024 * 1024 * 1024) },
    })
  }
  if (target === 'https://cdn.example/huge.png') {
    return new Response(PNG, {
      status: 200,
      headers: { 'content-type': 'image/png', 'content-length': String(200 * 1024 * 1024) },
    })
  }
  if (target === 'https://cdn.example/octet.bin') {
    return new Response(PNG, { status: 200, headers: { 'content-type': 'application/octet-stream' } })
  }
  if (target.startsWith('https://token-plan')) {
    const body = JSON.parse(String(init?.body))
    const prompt = body.input?.messages?.[0]?.content?.find((part) => typeof part.text === 'string')?.text ?? ''
    if (prompt === 'budget slow') return hangUntilAborted()
    if (prompt === 'bad size') {
      return failure(400, { code: 'InvalidParameter', message: 'size must be one of 1024*1024, 1536*1024', request_id: 'req-size' })
    }
    if (prompt === 'huge image') return json(imageBodyWith('https://cdn.example/huge.png'))
    if (prompt === 'empty result') return json({ output: { choices: [] }, request_id: 'req-empty' })
    if (prompt === 'data url') return json(imageBodyWith('data:image/png;base64,' + PNG.toString('base64')))
    if (prompt === 'octet image') return json(imageBodyWith('https://cdn.example/octet.bin'))
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
      return json(imageBodyWith('https://cdn.example/single-' + singleImageSerial + '.png'))
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

/** Register the plugin against the shared stub services and return its tools. */
const makePlugin = (overrides = {}, options = {}) => {
  const tools = {}
  const ctx = {
    effect: (fn) => fn(),
    tools: { register: (definition) => { tools[definition.name] = definition; return () => {} } },
    get: (name) => {
      if (name === 'credentials' && options.credentials !== false) {
        return { resolve: async () => ({ value: 'sk-sp-test', source: 'store' }) }
      }
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
  mod.apply(ctx, { videoPollIntervalMs: 50, videoWaitMs: 1000, ...overrides })
  return tools
}

/** Run one tool call and capture either its value or its failure. */
const attempt = async (tool, args, execOverride = exec) => {
  try {
    return { value: await tool.execute(args, execOverride) }
  } catch (error) {
    return { error }
  }
}

const registered = makePlugin()
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
expect('manifest records the record kind', manifestHit.kind, 'image')
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


// --- image tool: an unrelated 400 is not retried per image ----------------
calls.length = 0
const badSize = await attempt(registered.qwen_image_generate, { prompt: 'bad size', count: 3 })
expect('unrelated 400 is not retried per image', calls.filter((call) => String(call.url).includes('/multimodal-generation')).length, 1)
expect('unrelated 400 keeps the request failure code', badSize.error?.code, 'QWEN_IMAGE_REQUEST_FAILED')
expect('unrelated 400 surfaces the provider reason', String(badSize.error?.message).includes('size must be one of'), true)

// --- image tool: reserved device names never become file stems ------------
expect('reserved file stem rejected', await budgetRejected({ prompt: 'a cat', name_prefix: 'CON' }), 'INVALID_ARGS')

// --- image tool: data URLs and signature sniffing -------------------------
const dataUrlImage = await registered.qwen_image_generate.execute({ prompt: 'data url' }, exec)
expect('data URL image saved with its sniffed extension', dataUrlImage.images[0].path.endsWith('.png'), true)
const octetImage = await registered.qwen_image_generate.execute({ prompt: 'octet image' }, exec)
expect('media type falls back to the signature', octetImage.images[0].path.endsWith('.png'), true)

// --- image tool: oversized downloads are refused --------------------------
const hugeImage = await attempt(registered.qwen_image_generate, { prompt: 'huge image' })
expect('oversized image download is refused', String(hugeImage.error?.message).includes('maxImageBytes'), true)
expect('all-download failure names a source', String(hugeImage.error?.message).includes('cdn.example/huge.png'), true)

// --- image tool: a spent generation budget still saves finished images ----
const budgeted = makePlugin({ timeoutMs: 1500 })
const budgetOnly = await attempt(budgeted.qwen_image_generate, { prompt: 'budget slow', output_dir: 'budget-images' })
expect('image budget expiry gets its own code', budgetOnly.error?.code, 'QWEN_IMAGE_TIMEOUT')
expect('image budget message is actionable', String(budgetOnly.error?.message).includes('pass fewer prompts per call'), true)
const budgetPartial = await attempt(budgeted.qwen_image_generate, {
  prompts: ['a budget fast', 'budget slow'],
  output_dir: 'budget-images',
})
expect('spent generation budget still saves finished images', budgetPartial.value?.images?.length, 1)
expect('partial budget run reports the loss in warnings', Array.isArray(budgetPartial.value?.warnings)
  && budgetPartial.value.warnings.some((warning) => warning.includes('budget ran out')), true)
const budgetManifest = (await readFile(join(dir, 'budget-images', '.manifest.jsonl'), 'utf8'))
  .trim().split(String.fromCharCode(10)).map((line) => JSON.parse(line))
expect('budget run appends its manifest', budgetManifest.every((line) => line.kind === 'image'), true)

// --- image tool: a long negative_prompt is bounded and reported -----------
calls.length = 0
const longNegative = await registered.qwen_image_generate.execute({
  prompt: 'a cat',
  model: 'qwen-image-3.0-pro',
  negative_prompt: 'x'.repeat(600),
}, exec)
expect('trimmed negative_prompt is reported', Array.isArray(longNegative.warnings)
  && longNegative.warnings.some((warning) => warning.includes('cut to its first 500')), true)
const longNegativeBody = JSON.parse(String(calls.filter((call) => String(call.url).includes('/multimodal-generation')).pop().init.body))
expect('negative_prompt is bounded before sending', longNegativeBody.parameters.negative_prompt.length, 500)

// --- video tool: concurrent submissions keep every billed task id ---------
calls.length = 0
await Promise.all([
  registered.qwen_video_generate.execute({ prompt: 'hang one', wait_seconds: 1 }, exec),
  registered.qwen_video_generate.execute({ prompt: 'hang two', wait_seconds: 1 }, exec),
])
const ledgerAfter = JSON.parse(await readFile(join(dir, 'qwen-videos', '.pending-tasks.json'), 'utf8'))
expect('concurrent submissions keep every task id', [
  Object.hasOwn(ledgerAfter, 'task-pending-1'),
  Object.hasOwn(ledgerAfter, 'task-pending-2'),
], [true, true])

// --- video tool: status-less polls are tolerated, endless ones fail -------
calls.length = 0
unknownPolls = 0
const unknownThenOk = await registered.qwen_video_generate.execute({ prompt: 'unknown then ok', wait_seconds: 5 }, exec)
expect('status-less polls are tolerated', unknownThenOk.status, 'succeeded')
const alwaysUnknown = await attempt(registered.qwen_video_generate, { prompt: 'always unknown', wait_seconds: 5 })
expect('a status-less task stays pending', alwaysUnknown.value?.status, 'pending')
expect('status-less result explains how to resume', String(alwaysUnknown.value?.message).includes('task_id'), true)
const ledgerUnknown = JSON.parse(await readFile(join(dir, 'qwen-videos', '.pending-tasks.json'), 'utf8'))
expect('a status-less task keeps its billed id', Object.hasOwn(ledgerUnknown, 'task-dead'), true)

// --- video tool: a poll timeout returns pending, not retried extra -------
const impatientPoll = makePlugin({ videoWaitMs: 500 })
const slowPoll = await impatientPoll.qwen_video_generate.execute({ prompt: 'timeout query', wait_seconds: 1 }, exec)
expect('a poll timeout still returns the task id', slowPoll.status, 'pending')
expect('a poll timeout keeps its task id', slowPoll.taskId, 'task-timeout-query')

// --- video tool: transient query failures are retried ---------------------
calls.length = 0
flakyPolls = 0
const flaky = await registered.qwen_video_generate.execute({ prompt: 'flaky query', wait_seconds: 5 }, exec)
expect('transient query failures are retried', flaky.status, 'succeeded')
expect('query retry count matches the failures', flakyPolls, 3)

// --- image tool: a missing key and an empty answer are actionable ---------
const noKey = makePlugin({ apiKeyEnv: 'QWEN_TEST_MISSING_KEY' }, { credentials: false })
const noKeyResult = await attempt(noKey.qwen_image_generate, { prompt: 'a cat' })
expect('missing api key gets its own code', noKeyResult.error?.code, 'QWEN_IMAGE_NO_API_KEY')
expect('missing api key names the credential', String(noKeyResult.error?.message).includes('QWEN_TEST_MISSING_KEY'), true)
const emptyResult = await attempt(registered.qwen_image_generate, { prompt: 'empty result' })
expect('an answer with no image is actionable', emptyResult.error?.code, 'QWEN_IMAGE_NO_IMAGE')

// --- image tool: caller cancellation surfaces as cancellation -------------
const controller = new AbortController()
const abortPromise = attempt(registered.qwen_image_generate, { prompt: 'budget slow', output_dir: 'abort-images' }, {
  ...exec,
  signal: controller.signal,
})
setTimeout(() => controller.abort(), 50)
const cancelled = await abortPromise
expect('caller cancellation keeps the abort shape', cancelled.error?.name, 'AbortError')

// --- video tool: a resumable task reports its recorded model --------------
calls.length = 0
const primePending = await registered.qwen_video_generate.execute({
  prompt: 'hang forever',
  model: 'wan3.0-video-prime',
  wait_seconds: 1,
}, exec)
expect('prime pending submits the requested model', JSON.parse(calls.find((call) => call.url.endsWith('/video-synthesis')).init.body).model, 'wan3.0-video-prime')
const primeResume = await registered.qwen_video_generate.execute({ task_id: 'task-pending', prompt: 'ignored prompt', wait_seconds: 1 }, exec)
expect('resume reports the recorded model', primeResume.model, 'wan3.0-video-prime')
expect('resume warns that prompt was ignored', Array.isArray(primeResume.warnings)
  && primeResume.warnings.some((warning) => warning.includes('prompt/media were ignored')), true)

// --- video tool: the manifest records the finished render -----------------
const videoManifest = (await readFile(join(dir, 'qwen-videos', '.manifest.jsonl'), 'utf8'))
  .trim().split(String.fromCharCode(10)).map((line) => JSON.parse(line))
const videoRecord = videoManifest.find((line) => line.taskId === 'task-1')
expect('video manifest records the record kind', videoRecord?.kind, 'video')
expect('video manifest records the prompt', videoRecord?.prompt, 'a cat on a roof')
expect('video manifest records the saved file', typeof videoRecord?.file, 'string')

// --- video tool: an interrupted download keeps the task id ----------------
const impatient = makePlugin({ videoTimeoutMs: 400 })
calls.length = 0
const interrupted = await impatient.qwen_video_generate.execute({ task_id: 'task-download-slow', wait_seconds: 1 }, exec)
expect('interrupted video download stays pending', interrupted.status, 'pending')
expect('interrupted download explains how to resume', String(interrupted.message).includes('download was interrupted'), true)
const ledgerAfterInterrupt = JSON.parse(await readFile(join(dir, 'qwen-videos', '.pending-tasks.json'), 'utf8'))
expect('interrupted download keeps its task id', Object.hasOwn(ledgerAfterInterrupt, 'task-download-slow'), true)

// --- video tool: oversized download and malformed endpoint fail clearly ---
const hugeVideo = await attempt(registered.qwen_video_generate, { prompt: 'huge video', wait_seconds: 5 })
expect('oversized video download is refused', String(hugeVideo.error?.message).includes('maxVideoBytes'), true)
const badEndpoint = makePlugin({ videoBaseUrl: 'https://token-plan.example/api/v1/services/aigc/image-generation' })
calls.length = 0
const badEndpointResult = await attempt(badEndpoint.qwen_video_generate, { prompt: 'anything', wait_seconds: 1 })
expect('bad videoBaseUrl fails with a config code', badEndpointResult.error?.code, 'QWEN_VIDEO_BAD_CONFIG')
expect('bad videoBaseUrl makes no request', calls.length, 0)

// --- argument guards: output_dir must be a string -------------------------
expect('image output_dir must be a string', await budgetRejected({ prompt: 'a cat', output_dir: 42 }), 'INVALID_ARGS')
expect('video output_dir must be a string', (await attempt(registered.qwen_video_generate, { prompt: 'x', output_dir: 42 })).error?.code, 'INVALID_ARGS')

// --- output_dir cannot escape through a workspace link --------------------
let linkMade = false
try {
  await symlink(tmpdir(), join(dir, 'escape-link'), 'junction')
  linkMade = true
} catch {
  try {
    await symlink(tmpdir(), join(dir, 'escape-link'))
    linkMade = true
  } catch {
    linkMade = false
  }
}
if (linkMade) {
  const escapedLink = await attempt(registered.qwen_image_generate, { prompt: 'a cat', output_dir: 'escape-link/qwen' })
  expect('output_dir cannot escape through a workspace link', escapedLink.error?.code, 'INVALID_ARGS')
  await rm(join(dir, 'escape-link'), { recursive: true, force: true })
} else {
  console.log('SKIP output_dir link escape (symlinks unavailable)')
}

// --- video tool: a corrupt ledger is preserved, not silently erased -------
const corruptDir = join(dir, 'corrupt-ledger')
await mkdir(corruptDir, { recursive: true })
await writeFile(join(corruptDir, '.pending-tasks.json'), '{"task-broken": ')
const corruptRun = await registered.qwen_video_generate.execute({
  prompt: 'hang forever',
  output_dir: 'corrupt-ledger',
  wait_seconds: 1,
}, exec)
expect('corrupt ledger does not block a submission', corruptRun.status, 'pending')
const corruptFiles = await readdir(corruptDir)
expect('corrupt ledger is preserved for inspection', corruptFiles.some((name) => name.startsWith('.pending-tasks.json.corrupt-')), true)

// --- video tool: a submission timeout is unknown, not a network failure ---
const impatientSubmit = makePlugin({ videoTimeoutMs: 300 })
const slowSubmit = await attempt(impatientSubmit.qwen_video_generate, { prompt: 'slow submit', wait_seconds: 1 })
expect('submission timeout reports an unknown-state code', slowSubmit.error?.code, 'QWEN_VIDEO_SUBMIT_UNKNOWN')

// --- video tool: waiting past the host default is flagged -----------------
const overHostWait = await registered.qwen_video_generate.execute({ prompt: 'wait warning check', wait_seconds: 600 }, exec)
expect('over-host wait gets an advisory warning', Array.isArray(overHostWait.warnings)
  && overHostWait.warnings.some((warning) => warning.includes('wait_seconds')), true)

await rm(dir, { recursive: true, force: true })
if (failures.length > 0) {
  console.error('FAILURES: ' + failures.join(' | '))
  process.exitCode = 1
}
