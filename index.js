/**
 * Qwen Token Plan image and video generation for DeepSeek Harness.
 *
 * Registers two model-facing tools: `qwen_image_generate` calls the
 * multimodal-generation endpoint, downloads the generated PNG files into the
 * session workspace, and returns them both as local paths and (when the active
 * model route accepts images) as chat image blocks; `qwen_video_generate`
 * submits an asynchronous video-synthesis task, polls it, and downloads the
 * finished MP4 into the session workspace.
 *
 * Configuration arrives from the bundle patch row:
 * - `apiKeyEnv`  credential reference resolved through `ctx.credentials`, then
 *                the launch environment (default `DASHSCOPE_API_KEY`)
 * - `baseUrl`    generation endpoint
 * - `defaultModel` / `defaultSize` image tool defaults
 * - `defaultVideoModel` / `defaultVideoResolution` / `defaultVideoRatio` /
 *                `defaultVideoDuration` video tool defaults
 * - `videoBaseUrl` asynchronous video-synthesis endpoint (task URL derived)
 * - `videoTimeoutMs` / `videoPollIntervalMs` / `videoWaitMs` video budgets;
 *   `videoWaitMs` stays under the host's default per-call timeout so a poll
 *   window does not outlive the tool call that owns it
 * - `outputDir`  directory used when the session directory is unavailable
 * - `timeoutMs`  budget for one image call, split between requests and downloads
 * - `maxImageBytes` / `maxVideoBytes` download size caps
 *
 * The plugin imports no Harness package at runtime: profile-installed bundles
 * resolve Harness packages through the loader, and staying dependency-free keeps
 * activation robust.
 *
 * @module dsh-qwen-image
 */

import { appendFile, mkdir, readFile, realpath, rename, rm, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, join, relative as relativePath, resolve as resolvePath, sep } from 'node:path'

/** Cordis plugin name used by loader diagnostics. */
export const name = 'dsh-qwen-image'

/** Services this plugin uses; only the tool registry is mandatory. */
export const inject = ['tools']

const DEFAULT_API_KEY_ENV = 'DASHSCOPE_API_KEY'
const DEFAULT_BASE_URL = 'https://token-plan.maas.qianwenaiapi.com/api/v1/services/aigc/multimodal-generation/generation'
const DEFAULT_MODEL = 'wan2.7-image-pro'
const DEFAULT_VIDEO_MODEL = 'wan3.0-video'
const DEFAULT_VIDEO_BASE_URL = 'https://token-plan.maas.qianwenaiapi.com/api/v1/services/aigc/video-generation/video-synthesis'
const DEFAULT_VIDEO_RESOLUTION = '720P'
const DEFAULT_VIDEO_RATIO = '16:9'
const DEFAULT_VIDEO_DURATION = 5
const DEFAULT_VIDEO_TIMEOUT_MS = 180_000
const DEFAULT_VIDEO_POLL_INTERVAL_MS = 5_000
/**
 * Default poll window. Kept below the host's default 120 s per-call timeout: a
 * window longer than the host allows leaves the task running (and billed) while
 * the process is killed mid-poll and the task id is lost with it.
 */
const DEFAULT_VIDEO_WAIT_MS = 90_000
const DEFAULT_SIZE = '1024*1024'
/**
 * Whole-call budget for one image request batch. Kept under the host's default
 * 120 s per-call timeout: a budget the host outlives means the call is killed
 * before this plugin can report anything, and a synchronous image request has
 * no task id to resume, so a paid result is simply lost. A slice of it is
 * reserved for downloading images the generation phase already paid for.
 */
const DEFAULT_TIMEOUT_MS = 110_000
/**
 * Share of one image call's budget kept back for downloading generated files.
 * Result URLs expire and the images are already billed, so the fetch phase must
 * not lose the race against the generation budget.
 */
const DOWNLOAD_RESERVE_MS = 15_000
/** Floor for the image download phase, even when generation spent the budget. */
const DOWNLOAD_FLOOR_MS = 5_000
/** Ceiling for the image download phase, so one call stays near the budget. */
const DOWNLOAD_CEILING_MS = 30_000
/** Per-request budget for fetching a finished video; the task id survives a failure. */
const VIDEO_DOWNLOAD_BUDGET_MS = 60_000
/** Attempts (including the first) for one retryable video download. */
const MAX_VIDEO_DOWNLOAD_ATTEMPTS = 3
/** Download caps: one runaway response must not exhaust memory. */
const DEFAULT_MAX_IMAGE_BYTES = 64 * 1024 * 1024
const DEFAULT_MAX_VIDEO_BYTES = 512 * 1024 * 1024
/** Submission budget: creating a task is quick and must not outlive the host's kill window. */
const VIDEO_SUBMIT_BUDGET_MS = 60_000
/** The host's default per-call timeout, used only for an advisory warning. */
const HOST_DEFAULT_TIMEOUT_MS = 120_000
/** Hard cap on images one call may produce, whatever the parameter shape. */
const MAX_IMAGES_PER_CALL = 6
/** Attempts (including the first) for one retryable image request or download. */
const MAX_IMAGE_ATTEMPTS = 3
/** First retry delay; doubles per attempt and honours a Retry-After hint. */
const IMAGE_RETRY_DELAY_MS = 800
/** How many prompt requests one batch call may keep in flight. */
const IMAGE_BATCH_CONCURRENCY = 3
/** JSONL ledger of generated files, appended in the output directory. */
const MANIFEST_FILE = '.manifest.jsonl'

/** Models documented for the Token Plan image endpoint. */
const KNOWN_MODELS = [
  'qwen-image-3.0-pro',
  'qwen-image-2.0-pro',
  'qwen-image-2.0',
  'wan2.7-image',
  'wan2.7-image-pro',
]

/** Models documented for the Token Plan video endpoint (Wan 3.0 and HappyHorse). */
const KNOWN_VIDEO_MODELS = [
  'wan3.0-video',
  'wan3.0-video-prime',
  'happyhorse-1.1-t2v',
  'happyhorse-1.1-i2v',
  'happyhorse-1.1-r2v',
]

/** Resolution tiers shared by the documented video models. */
const VIDEO_RESOLUTIONS = ['480P', '720P', '1080P']

/** Aspect ratios across the documented video models; `adaptive` lets Wan choose. */
const VIDEO_RATIOS = ['adaptive', '21:9', '16:9', '4:3', '1:1', '3:4', '9:16', '4:5', '5:4', '9:21']

/** Media slot types understood by `wan3.0-video` input.media. */
const VIDEO_MEDIA_TYPES = [
  'first_frame',
  'last_frame',
  'reference_image',
  'reference_video',
  'reference_audio',
  'file',
  'link',
]

/** Path fragment that identifies the asynchronous video-synthesis endpoint. */
const VIDEO_SUBMIT_SUFFIX = '/services/aigc/video-generation/video-synthesis'

/** Ledger of submitted-but-unfinished tasks, kept beside the downloaded videos. */
const PENDING_TASKS_FILE = '.pending-tasks.json'

/** Consecutive retryable poll failures tolerated before the wait gives up. */
const MAX_QUERY_FAILURES = 5

/** Consecutive status-less poll answers tolerated before a task is called dead. */
const MAX_UNKNOWN_POLLS = 6

/** Cap on one retry wait, whatever the service hints. */
const MAX_RETRY_DELAY_MS = 30_000

/** Resolved plugin configuration after defaults. */
function resolveConfig(raw) {
  const config = raw !== null && typeof raw === 'object' ? raw : {}
  const positive = (value, fallback) => (Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback)
  const text = (value, fallback) => (typeof value === 'string' && value.trim() !== '' ? value.trim() : fallback)
  return {
    apiKeyEnv: text(config.apiKeyEnv, DEFAULT_API_KEY_ENV),
    baseUrl: text(config.baseUrl, DEFAULT_BASE_URL),
    defaultModel: text(config.defaultModel, DEFAULT_MODEL),
    defaultSize: text(config.defaultSize, DEFAULT_SIZE),
    videoBaseUrl: text(config.videoBaseUrl, DEFAULT_VIDEO_BASE_URL),
    defaultVideoModel: text(config.defaultVideoModel, DEFAULT_VIDEO_MODEL),
    defaultVideoResolution: text(config.defaultVideoResolution, DEFAULT_VIDEO_RESOLUTION),
    defaultVideoRatio: text(config.defaultVideoRatio, DEFAULT_VIDEO_RATIO),
    defaultVideoDuration: positive(config.defaultVideoDuration, DEFAULT_VIDEO_DURATION),
    videoTimeoutMs: positive(config.videoTimeoutMs, DEFAULT_VIDEO_TIMEOUT_MS),
    videoPollIntervalMs: positive(config.videoPollIntervalMs, DEFAULT_VIDEO_POLL_INTERVAL_MS),
    videoWaitMs: positive(config.videoWaitMs, DEFAULT_VIDEO_WAIT_MS),
    outputDir: text(config.outputDir, ''),
    timeoutMs: positive(config.timeoutMs, DEFAULT_TIMEOUT_MS),
    maxImageBytes: positive(config.maxImageBytes, DEFAULT_MAX_IMAGE_BYTES),
    maxVideoBytes: positive(config.maxVideoBytes, DEFAULT_MAX_VIDEO_BYTES),
  }
}

/** A failure the model can act on, carrying a stable code for the tool result. */
class QwenImageError extends Error {
  constructor(message, code, status) {
    super(message)
    this.name = 'QwenImageError'
    this.code = code
    /** HTTP status when the failure came from a response; drives retry choices. */
    this.status = status
  }
}

/** A harness tool argument failure: the model must change the arguments. */
class ImageArgsError extends Error {
  constructor(message) {
    super(message)
    this.name = 'ImageArgsError'
    this.code = 'INVALID_ARGS'
  }
}

/** Whether a value is a non-blank string. */
function isNonBlankString(value) {
  return typeof value === 'string' && value.trim() !== ''
}

/**
 * Read the plan API key. The credential store wins, then the launch environment;
 * the reference name is what configuration holds, never the value itself.
 */
async function resolveApiKey(ctx, apiKeyEnv) {
  const credentials = ctx.get('credentials')
  if (credentials !== undefined) {
    try {
      const hit = await credentials.resolve(apiKeyEnv)
      if (hit !== undefined && isNonBlankString(hit.value)) return hit.value
    } catch {
      // A refused reference is not fatal: the environment layer may still hold it.
    }
  }
  const fromEnv = process.env[apiKeyEnv]
  if (isNonBlankString(fromEnv)) return fromEnv.trim()
  throw new QwenImageError(
    `no Qwen Token Plan API key: store it as the credential "${apiKeyEnv}" `
      + `(Settings > Models and credentials, or the credentials file) or export ${apiKeyEnv} before launching DSH`,
    'QWEN_IMAGE_NO_API_KEY',
  )
}

/** Extract every generated image reference from one successful response body. */
function imageEntriesOf(body) {
  const choices = body?.output?.choices
  if (!Array.isArray(choices)) return { images: [], rewriteStatus: undefined }
  const images = []
  for (const choice of choices) {
    const content = choice?.message?.content
    if (!Array.isArray(content)) continue
    for (const part of content) {
      if (typeof part?.image === 'string' && part.image.trim() !== '') images.push(part.image.trim())
    }
  }
  return { images, rewriteStatus: typeof body?.output?.rewrite_status === 'string' ? body.output.rewrite_status : undefined }
}

/** Read a provider failure body as one actionable sentence, request id included. */
function providerMessage(text, status) {
  const trimmed = text.trim()
  if (trimmed === '') return `HTTP ${status}`
  try {
    const parsed = JSON.parse(trimmed)
    const message = parsed?.message ?? parsed?.error?.message ?? parsed?.code
    const requestId = parsed?.request_id ?? parsed?.error?.request_id
    const suffix = isNonBlankString(requestId) ? ` (request_id ${String(requestId).trim()})` : ''
    if (isNonBlankString(message)) return `HTTP ${status}: ${message.trim()}${suffix}`
    return `HTTP ${status}: ${trimmed.slice(0, 400)}${suffix}`
  } catch {
    // Not JSON: keep the bounded raw text.
  }
  return `HTTP ${status}: ${trimmed.slice(0, 400)}`
}

/** Call the generation endpoint and return the parsed body. */
async function requestGeneration(options) {
  const { baseUrl, apiKey, model, prompt, size, count, negativePrompt, seed, watermark, promptExtend, images, signal } = options
  const content = []
  for (const image of images ?? []) content.push({ image })
  content.push({ text: prompt })
  const parameters = {}
  if (size !== undefined) parameters.size = size
  // Only state a multi-image count: the field's name and support differ across
  // model families, and one image is every model's default anyway.
  if (count !== undefined && count > 1) parameters.n = count
  // Negative prompts are a qwen-image-3.0-series parameter; other models reject
  // unknown parameters instead of ignoring them.
  if (negativePrompt !== undefined && model.startsWith('qwen-image-3.')) parameters.negative_prompt = negativePrompt
  if (seed !== undefined) parameters.seed = seed
  if (watermark !== undefined) parameters.watermark = watermark
  if (promptExtend !== undefined) parameters.prompt_extend = promptExtend

  let response
  try {
    response = await fetch(baseUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        input: { messages: [{ role: 'user', content }] },
        parameters,
      }),
      signal,
    })
  } catch (error) {
    // A timeout leaves it unknown whether the request was billed, so it is
    // surfaced as a cancellation instead of being retried and billed twice.
    if (isAbortError(error)) throw error
    throw new QwenImageError(`the image request failed before a response: ${error?.message ?? String(error)}`, 'QWEN_IMAGE_NETWORK')
  }

  const text = await response.text()
  if (!response.ok) {
    const detail = providerMessage(text, response.status)
    const hint = response.status === 401 || response.status === 403
      ? ' — the API key is missing, expired, or not authorized for this model'
      : response.status === 429
        ? ' — the plan rate limit was reached; retry later'
        : response.status === 400
          ? ' — the model, size, or prompt was rejected'
          : ''
    throw new QwenImageError(`the image request was refused: ${detail}${hint}`, 'QWEN_IMAGE_REQUEST_FAILED', response.status)
  }
  try {
    return JSON.parse(text)
  } catch {
    throw new QwenImageError(`the image response was not JSON: ${text.trim().slice(0, 200)}`, 'QWEN_IMAGE_BAD_RESPONSE')
  }
}

/** One actionable failure for a download that exceeds its configured cap. */
function oversizeError(what, bytes, limit, configKey, code) {
  return new QwenImageError(
    what + ' is ' + bytes + ' bytes, over the ' + limit + '-byte cap; raise ' + configKey + ' to accept it',
    code,
  )
}

/**
 * Read a response body while never buffering more than `limitBytes`. A stated
 * Content-Length over the cap is refused before the body is read; otherwise the
 * stream is abandoned as soon as the accumulated size crosses the cap, so a
 * mislabeled or endless response cannot exhaust memory.
 */
async function readBodyLimited(response, limitBytes, what, configKey, code) {
  const stated = Number(response.headers.get('content-length'))
  if (Number.isFinite(stated) && stated > limitBytes) throw oversizeError(what, stated, limitBytes, configKey, code)
  if (response.body === null) {
    const buffer = new Uint8Array(await response.arrayBuffer())
    if (buffer.byteLength > limitBytes) throw oversizeError(what, buffer.byteLength, limitBytes, configKey, code)
    return buffer
  }
  const reader = response.body.getReader()
  const chunks = []
  let total = 0
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      total += value.byteLength
      if (total > limitBytes) {
        await reader.cancel().catch(() => {})
        throw oversizeError(what, total, limitBytes, configKey, code)
      }
      chunks.push(value)
    }
  } finally {
    reader.releaseLock()
  }
  const data = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) {
    data.set(chunk, offset)
    offset += chunk.byteLength
  }
  return data
}

/** Download one produced image into bytes plus its media type. */
async function downloadImage(url, signal, maxBytes) {
  if (url.startsWith('data:')) {
    const comma = url.indexOf(',')
    if (comma === -1) throw new QwenImageError('the returned data URL carried no payload', 'QWEN_IMAGE_BAD_RESPONSE')
    const header = url.slice(5, comma)
    if (!header.split(';').map((part) => part.trim().toLowerCase()).includes('base64')) {
      throw new QwenImageError('the returned data URL is not base64-encoded', 'QWEN_IMAGE_BAD_RESPONSE')
    }
    const data = new Uint8Array(Buffer.from(url.slice(comma + 1), 'base64'))
    if (data.byteLength > maxBytes) {
      throw oversizeError('the returned data URL image', data.byteLength, maxBytes, 'maxImageBytes', 'QWEN_IMAGE_DOWNLOAD_TOO_LARGE')
    }
    // The media type declared by a data URL is a hint; the signature decides.
    const mediaType = sniffMediaType(data)
    if (mediaType === undefined) {
      throw new QwenImageError('the returned data URL is not a PNG/JPEG/WebP/GIF image', 'QWEN_IMAGE_BAD_RESPONSE')
    }
    return { data, mediaType }
  }
  const response = await fetch(url, { signal })
  if (!response.ok) {
    throw new QwenImageError(
      'downloading the generated image failed with HTTP ' + response.status,
      'QWEN_IMAGE_DOWNLOAD_FAILED',
      response.status,
    )
  }
  const buffer = await readBodyLimited(response, maxBytes, 'the generated image', 'maxImageBytes', 'QWEN_IMAGE_DOWNLOAD_TOO_LARGE')
  const header = (response.headers.get('content-type') ?? '').split(';')[0]?.trim().toLowerCase()
  const mediaType = header === 'image/jpeg' || header === 'image/webp' || header === 'image/gif' || header === 'image/png'
    ? header
    : sniffMediaType(buffer)
  if (mediaType === undefined) {
    throw new QwenImageError('the downloaded file is not a PNG/JPEG/WebP/GIF image', 'QWEN_IMAGE_DOWNLOAD_FAILED')
  }
  return { data: buffer, mediaType }
}

/** Identify a supported image container from its signature bytes. */
function sniffMediaType(data) {
  const at = (offset, bytes) => bytes.every((byte, index) => data[offset + index] === byte)
  if (at(0, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) return 'image/png'
  if (at(0, [0xff, 0xd8, 0xff])) return 'image/jpeg'
  if (at(0, [0x47, 0x49, 0x46, 0x38])) return 'image/gif'
  if (at(0, [0x52, 0x49, 0x46, 0x46]) && at(8, [0x57, 0x45, 0x42, 0x50])) return 'image/webp'
  return undefined
}

/** File extension for one accepted image media type. */
function extensionOf(mediaType) {
  return mediaType === 'image/jpeg' ? '.jpg' : mediaType === 'image/webp' ? '.webp' : mediaType === 'image/gif' ? '.gif' : '.png'
}

/**
 * Write bytes to a free file name. The timestamp only resolves to the second, so
 * two calls that start together would otherwise overwrite each other; the
 * exclusive create makes the choice atomic and a collision moves to the next
 * suffix instead of clobbering the first file.
 */
async function writeUniqueFile(directory, stem, extension, data) {
  for (let attempt = 1; attempt <= 50; attempt += 1) {
    const path = join(directory, attempt === 1 ? stem + extension : stem + '-' + attempt + extension)
    try {
      await writeFile(path, data, { flag: 'wx' })
      return path
    } catch (error) {
      if (error?.code !== 'EEXIST') throw error
    }
  }
  throw new QwenImageError('could not find a free file name in ' + directory, 'QWEN_IMAGE_NAME_CONFLICT')
}

/**
 * Append one batch of records to the output directory's JSONL manifest. One
 * append per batch keeps concurrent calls from interleaving their lines.
 */
async function appendManifest(directory, entries) {
  const file = join(directory, MANIFEST_FILE)
  await appendFile(file, entries.map((entry) => JSON.stringify(entry)).join(String.fromCharCode(10)) + String.fromCharCode(10))
  return file
}

/** Timestamp fragment shared by one batch so its files sort together. */
function stamp(date = new Date()) {
  const pad = (value) => String(value).padStart(2, '0')
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}_${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`
}

/**
 * Pick the directory generated files land in: an explicit argument, then the
 * configured `outputDir`, then `leaf` under the session directory.
 */
async function outputDirectory(ctx, configured, requested, exec, leaf) {
  const session = await sessionDirectory(ctx, exec)
  const resolveAgainstSession = (value) => (isAbsolute(value) ? value : resolvePath(session, value))
  const explicit = isNonBlankString(requested) ? requested.trim() : undefined
  if (explicit !== undefined) {
    const target = resolveAgainstSession(explicit)
    // The model chooses this directory, so keep it inside the workspace: a
    // stray absolute path — or a link inside the workspace pointing out of it —
    // must not drop generated files into the system tree.
    if (!await resolvesInside(session, target)) {
      throw new ImageArgsError(`output_dir must stay inside the session workspace (${session}); got "${explicit}"`)
    }
    return target
  }
  const fallback = isNonBlankString(configured) ? configured.trim() : undefined
  return fallback === undefined ? join(session, leaf) : resolveAgainstSession(fallback)
}

/** Whether `candidate` stays under `root`, across separators and letter case. */
function isInside(root, candidate) {
  const step = relativePath(resolvePath(root), resolvePath(candidate))
  if (step === '') return true
  if (isAbsolute(step)) return false
  return step.split(sep)[0] !== '..'
}

/** The real path of the nearest existing ancestor, or undefined when unreachable. */
async function realAncestor(path) {
  let current = resolvePath(path)
  for (let step = 0; step < 64; step += 1) {
    try {
      return await realpath(current)
    } catch (error) {
      if (error?.code !== 'ENOENT' && error?.code !== 'ENOTDIR') return undefined
      const parent = dirname(current)
      if (parent === current) return undefined
      current = parent
    }
  }
  return undefined
}

/**
 * Whether `candidate` is lexically inside `root` and still inside it after
 * every existing symlink/junction on the way is resolved: a link planted in the
 * workspace must not redirect generated files outside it.
 */
async function resolvesInside(root, candidate) {
  if (!isInside(root, candidate)) return false
  const [realRoot, realCandidate] = await Promise.all([realAncestor(root), realAncestor(candidate)])
  // An unreachable ancestor leaves only the lexical check to go on.
  if (realRoot === undefined || realCandidate === undefined) return true
  return isInside(realRoot, realCandidate)
}

/** Resolve the session's working directory without failing the tool when absent. */
async function sessionDirectory(ctx, exec) {
  const workingDirectory = ctx.get('workingDirectory')
  if (workingDirectory !== undefined && exec.agent !== undefined) {
    try {
      return await workingDirectory.ensure(exec.agent, exec.signal)
    } catch {
      // Fall through to the recorded header and process defaults.
    }
  }
  const recorded = exec.agent?.session?.header?.cwd
  return isNonBlankString(recorded) ? recorded : process.cwd()
}

/** Whether the calling model route accepts image content. */
async function routeAcceptsImages(ctx, exec) {
  const llm = ctx.get('llm')
  if (llm === undefined || exec.agent === undefined) return false
  const routed = exec.agent.session?.requestHeader?.()?.config
  const provider = routed?.provider ?? exec.agent.options?.provider
  const model = routed?.model ?? exec.agent.options?.model
  if (provider === undefined || model === undefined) return false
  try {
    const info = await llm.resolveModelInfo(provider, model, exec.signal)
    return Array.isArray(info?.inputModalities) && info.inputModalities.includes('image')
  } catch {
    return false
  }
}

/** Combine the caller's cancellation with the request budget. */
function combineSignals(caller, timeout) {
  if (caller === undefined) return timeout
  if (typeof AbortSignal.any === 'function') return AbortSignal.any([caller, timeout])
  const controller = new AbortController()
  const abort = () => controller.abort()
  if (caller.aborted || timeout.aborted) {
    controller.abort()
    return controller.signal
  }
  caller.addEventListener('abort', abort, { once: true })
  timeout.addEventListener('abort', abort, { once: true })
  return controller.signal
}

/** A bounded, single-line rendering of one remote source for a message. */
function shortSource(source) {
  const text = String(source ?? '')
  return text.length <= 120 ? text : text.slice(0, 117) + '...'
}

/** The file-name part of a path, whatever the platform separator. */
function baseNameOf(path) {
  const separator = String.fromCharCode(92)
  const cut = Math.max(path.lastIndexOf('/'), path.lastIndexOf(separator))
  return cut === -1 ? path : path.slice(cut + 1)
}

/** Whether a failed image step may be retried instead of surfaced. */
function isTransientImageError(error) {
  if (error?.code === 'QWEN_IMAGE_NETWORK') return true
  if (error?.code !== 'QWEN_IMAGE_REQUEST_FAILED' && error?.code !== 'QWEN_IMAGE_DOWNLOAD_FAILED') return false
  return error.status === 429 || (Number.isInteger(error.status) && error.status >= 500)
}

/** Wait before one image retry, honouring a Retry-After hint. */
function imageRetryDelay(error, attempt) {
  const hinted = Number(error?.retryAfterMs)
  if (Number.isFinite(hinted) && hinted > 0) return Math.min(hinted, MAX_RETRY_DELAY_MS)
  return Math.min(IMAGE_RETRY_DELAY_MS * 2 ** (attempt - 1), MAX_RETRY_DELAY_MS)
}

/**
 * Run one image request or download with bounded retries. Image calls are the
 * expensive half of this plugin, so a dropped connection must not throw away a
 * paid result: retryable failures are retried here instead of reaching the
 * model, while cancellation still wins because the wait honours the signal.
 */
async function withImageRetry(run, signal) {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await run()
    } catch (error) {
      if (isAbortError(error) || !isTransientImageError(error) || attempt >= MAX_IMAGE_ATTEMPTS) throw error
      await sleep(imageRetryDelay(error, attempt), signal)
    }
  }
}

/**
 * Whether a 400 means the model rejected the batch count field. A prompt that
 * the provider merely refuses must never be retried once per image: retrying
 * costs one paid request per image for an answer that cannot change, so the
 * fallback only triggers when the failure names the count field itself.
 */
function isBatchCountRejected(error) {
  if (error?.status !== 400) return false
  const message = String(error?.message ?? '').toLowerCase()
  // The count field is reported as `n`, "batch", or "count"; a generic
  // "InvalidParameter" (a rejected size or seed, say) must not match, or a
  // request that can never succeed would be billed once per image.
  const namesCount = /(^|[^a-z])(?:n|batch|count)([^a-z]|$)/u.test(message)
    || /number\s+of\s+images?|image\s+count|batch\s+size/u.test(message)
  if (!namesCount) return false
  return /not\s+support|unsupport|invalid|unknown|unexpected|illegal|exceed|maximum|must\s+be|out\s+of\s+range|too\s+large/u.test(message)
}

/** One file-stem component: no separators, no reserved names, bounded length. */
function sanitizeFileStem(value) {
  const illegal = new Set(['/', ':', '*', '?', '"', '<', '>', '|', String.fromCharCode(92)])
  let cleaned = ''
  for (const character of value.normalize('NFC')) {
    const code = character.codePointAt(0)
    if (code < 32 || illegal.has(character) || character.trim() === '') cleaned += '_'
    else cleaned += character
  }
  cleaned = cleaned.replace(/^[._]+/u, '').replace(/[._]+$/u, '').slice(0, 60)
  if (cleaned === '' || cleaned === '.' || cleaned === '..') return undefined
  // Windows device names can never be a file stem; reject them here instead of
  // letting the create fail later with a platform-specific error.
  if (/^(?:con|prn|aux|nul|com[0-9]|lpt[0-9])$/iu.test(cleaned)) return undefined
  return cleaned
}

/** Map with a concurrency cap, preserving order and collecting each outcome. */
async function mapSettled(items, limit, run) {
  const outcomes = new Array(items.length)
  let next = 0
  const worker = async () => {
    for (;;) {
      const index = next
      next += 1
      if (index >= items.length) return
      try {
        outcomes[index] = { status: 'fulfilled', value: await run(items[index], index) }
      } catch (error) {
        outcomes[index] = { status: 'rejected', reason: error }
      }
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker))
  return outcomes
}

/** Validate and normalize the model-supplied arguments. */
function parseArgs(args, config) {
  if (args === null || typeof args !== 'object') throw new ImageArgsError('arguments must be an object')
  if (args.output_dir !== undefined && typeof args.output_dir !== 'string') {
    throw new ImageArgsError('output_dir must be a string')
  }
  const prompt = typeof args.prompt === 'string' ? args.prompt.trim() : ''
  const rawPrompts = args.prompts
  let prompts
  if (rawPrompts !== undefined) {
    if (!Array.isArray(rawPrompts) || rawPrompts.length === 0 || rawPrompts.length > MAX_IMAGES_PER_CALL) {
      throw new ImageArgsError('prompts must be an array of 1 to ' + MAX_IMAGES_PER_CALL + ' non-empty strings')
    }
    prompts = rawPrompts.map((entry, index) => {
      if (!isNonBlankString(entry)) throw new ImageArgsError('prompts[' + index + '] must be a non-empty string')
      return entry.trim()
    })
    if (prompt !== '') throw new ImageArgsError('pass either prompt (one string) or prompts (a list of them), not both')
  } else if (prompt === '') {
    throw new ImageArgsError('prompt must be a non-empty string (pass prompts to draw several different images in one call)')
  }

  const model = isNonBlankString(args.model) ? args.model.trim() : config.defaultModel
  if (!KNOWN_MODELS.includes(model)) {
    throw new ImageArgsError('model "' + model + '" is not a Token Plan image model; use one of ' + KNOWN_MODELS.join(', '))
  }
  const size = isNonBlankString(args.size) ? args.size.trim() : config.defaultSize
  if (!/^[0-9]{2,5}[*][0-9]{2,5}$/u.test(size)) {
    throw new ImageArgsError('size "' + size + '" must look like 1024*1024 (width*height)')
  }
  const count = args.count === undefined ? 1 : args.count
  if (!Number.isInteger(count) || count < 1 || count > MAX_IMAGES_PER_CALL) {
    throw new ImageArgsError('count must be an integer from 1 to ' + MAX_IMAGES_PER_CALL)
  }
  const promptCount = prompts === undefined ? 1 : prompts.length
  const total = promptCount * count
  if (total > MAX_IMAGES_PER_CALL) {
    throw new ImageArgsError('one call may produce at most ' + MAX_IMAGES_PER_CALL + ' images: ' + promptCount + ' prompt(s) x ' + count + ' asks for ' + total)
  }
  const sourceNames = args.name_prefix
  let namePrefixes
  if (sourceNames !== undefined) {
    const list = Array.isArray(sourceNames) ? sourceNames : [sourceNames]
    if (list.some((entry) => !isNonBlankString(entry))) {
      throw new ImageArgsError('name_prefix must be a non-empty string or an array of them')
    }
    if (Array.isArray(sourceNames) && sourceNames.length !== total) {
      throw new ImageArgsError('name_prefix as an array needs one name per image (' + total + '), got ' + sourceNames.length)
    }
    namePrefixes = list.map((entry) => {
      const stem = sanitizeFileStem(entry.trim())
      if (stem === undefined) throw new ImageArgsError('name_prefix "' + entry + '" has no usable file-name characters')
      return stem
    })
  }
  const seed = args.seed
  if (seed !== undefined && (!Number.isInteger(seed) || seed < 0 || seed > 2_147_483_647)) {
    throw new ImageArgsError('seed must be an integer from 0 to 2147483647')
  }
  const negativePrompt = args.negative_prompt
  if (negativePrompt !== undefined && typeof negativePrompt !== 'string') {
    throw new ImageArgsError('negative_prompt must be a string')
  }
  const trimmedNegative = typeof negativePrompt === 'string' ? negativePrompt.trim() : ''
  const images = args.image_urls
  if (images !== undefined) {
    if (!Array.isArray(images) || images.length > 3 || images.some((entry) => !isNonBlankString(entry))) {
      throw new ImageArgsError('image_urls must be an array of 1 to 3 non-empty image URLs or data URLs')
    }
  }
  return {
    prompt,
    prompts,
    namePrefixes,
    namePrefixIsList: Array.isArray(sourceNames),
    model,
    size,
    count,
    seed,
    negativePrompt: trimmedNegative === '' ? undefined : trimmedNegative.slice(0, 500),
    negativePromptWasTrimmed: trimmedNegative.length > 500,
    watermark: typeof args.watermark === 'boolean' ? args.watermark : undefined,
    promptExtend: typeof args.prompt_extend === 'boolean' ? args.prompt_extend : undefined,
    images: images?.map((entry) => entry.trim()),
  }
}

/** Derive the asynchronous task-query URL from the configured submit endpoint. */
function videoTaskUrl(baseUrl, taskId) {
  const at = baseUrl.indexOf(VIDEO_SUBMIT_SUFFIX)
  if (at === -1) {
    throw new QwenImageError(
      'videoBaseUrl must end with ' + VIDEO_SUBMIT_SUFFIX + ' so the task query URL can be derived; fix the plugin config',
      'QWEN_VIDEO_BAD_CONFIG',
    )
  }
  return baseUrl.slice(0, at) + '/tasks/' + encodeURIComponent(taskId)
}

/** Read the task status out of a submission or query body. */
function videoTaskStatus(body) {
  const status = body?.output?.task_status
  return isNonBlankString(status) ? status.trim().toUpperCase() : 'UNKNOWN'
}

/** One actionable sentence for a failed or cancelled video task. */
function videoTaskFailure(body) {
  const message = body?.output?.message ?? body?.message
  const code = body?.output?.code ?? body?.code
  const parts = []
  if (isNonBlankString(code)) parts.push(String(code).trim())
  if (isNonBlankString(message)) parts.push(String(message).trim())
  return parts.length === 0 ? 'the service reported no reason' : parts.join(': ')
}

/** Parse one JSON response body or fail with a stable code. */
function parseJsonBody(text, what) {
  try {
    return JSON.parse(text)
  } catch {
    throw new QwenImageError(what + ' response was not JSON: ' + text.trim().slice(0, 200), 'QWEN_VIDEO_BAD_RESPONSE')
  }
}

/** Submit one asynchronous video task and return its identifier. */
async function submitVideoTask(options) {
  const { baseUrl, apiKey, model, input, parameters, signal } = options
  let response
  try {
    response = await fetch(baseUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + apiKey,
        'X-DashScope-Async': 'enable',
      },
      body: JSON.stringify({ model, input, parameters }),
      signal,
    })
  } catch (error) {
    if (isAbortError(error)) throw error
    throw new QwenImageError('the video request failed before a response: ' + (error?.message ?? String(error)), 'QWEN_VIDEO_NETWORK')
  }
  const text = await response.text()
  if (!response.ok) {
    const hint = response.status === 401 || response.status === 403
      ? ' — the API key is missing, expired, or not authorized for this model'
      : response.status === 429
        ? ' — the plan rate limit was reached; retry later'
        : response.status === 400
          ? ' — the model, prompt, media, or parameters were rejected'
          : ''
    throw new QwenImageError('the video request was refused: ' + providerMessage(text, response.status) + hint, 'QWEN_VIDEO_REQUEST_FAILED')
  }
  const body = parseJsonBody(text, 'the video submission')
  const taskId = body?.output?.task_id
  if (!isNonBlankString(taskId)) {
    throw new QwenImageError('the service accepted the video request without a task id', 'QWEN_VIDEO_BAD_RESPONSE')
  }
  return String(taskId).trim()
}

/** Query one asynchronous video task and return its parsed body. */
async function queryVideoTask(url, apiKey, signal) {
  let response
  try {
    response = await fetch(url, { headers: { Authorization: 'Bearer ' + apiKey }, signal })
  } catch (error) {
    // Abort and timeout both mean the poll did not answer in time; treating the
    // timeout as a network failure would retry it and overshoot the wait window.
    if (isAbortError(error)) throw error
    throw new QwenImageError('querying the video task failed before a response: ' + (error?.message ?? String(error)), 'QWEN_VIDEO_NETWORK')
  }
  const text = await response.text()
  if (!response.ok) {
    const error = new QwenImageError(
      'querying the video task was refused: ' + providerMessage(text, response.status),
      'QWEN_VIDEO_QUERY_FAILED',
      response.status,
    )
    const retryAfter = Number(response.headers.get('retry-after'))
    if (Number.isFinite(retryAfter) && retryAfter > 0) error.retryAfterMs = retryAfter * 1000
    throw error
  }
  return parseJsonBody(text, 'the video task')
}

/** Extract the produced video URL from a finished task body. */
function videoSourceOf(body) {
  const output = body?.output ?? {}
  if (isNonBlankString(output.video_url)) return output.video_url.trim()
  const results = output.video_results
  if (Array.isArray(results)) {
    for (const entry of results) {
      if (isNonBlankString(entry?.url)) return entry.url.trim()
    }
  }
  return undefined
}

/** Per-directory promise chains that serialize ledger read-modify-write. */
const ledgerLocks = new Map()

/**
 * Serialize one ledger update per output directory. Two concurrent video calls
 * would otherwise both read the old file and the later write would drop the
 * other call's task id — the id of a render that was already billed.
 */
function withLedgerLock(directory, run) {
  const key = resolvePath(directory)
  const previous = ledgerLocks.get(key) ?? Promise.resolve()
  const result = previous.then(run, run)
  const settled = result.catch(() => {})
  ledgerLocks.set(key, settled)
  return result.finally(() => {
    if (ledgerLocks.get(key) === settled) ledgerLocks.delete(key)
  })
}

/**
 * Read the pending-task ledger. A missing file reads as an empty ledger: a
 * request that already cost quota must never be blocked by bookkeeping. A
 * corrupt file is moved aside first, so the next update cannot silently erase
 * the only trace of task ids that may still be rendering.
 */
async function readPendingTasks(directory) {
  const file = join(directory, PENDING_TASKS_FILE)
  try {
    const parsed = JSON.parse(await readFile(file, 'utf8'))
    return parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}
  } catch (error) {
    if (error?.code !== 'ENOENT') {
      await rename(file, file + '.corrupt-' + Date.now()).catch(() => {})
    }
    return {}
  }
}

/** Persist the pending-task ledger, dropping the file when nothing is pending. */
async function writePendingTasks(directory, tasks) {
  const file = join(directory, PENDING_TASKS_FILE)
  if (Object.keys(tasks).length === 0) {
    await rm(file, { force: true })
    return
  }
  // Write beside the ledger and rename over it: a process killed mid-write must
  // never leave a truncated ledger that the next update would overwrite with a
  // single task, losing every other task id.
  const temporary = file + '.' + process.pid + '-' + Date.now() + '.tmp'
  await writeFile(temporary, JSON.stringify(tasks, null, 2) + '\n')
  try {
    await rename(temporary, file)
  } catch (error) {
    await rm(temporary, { force: true }).catch(() => {})
    throw error
  }
}

/** Record one submitted task so a later call can resume it after a crash. */
async function rememberPendingTask(directory, taskId, record) {
  try {
    await withLedgerLock(directory, async () => {
      const tasks = await readPendingTasks(directory)
      tasks[taskId] = record
      await writePendingTasks(directory, tasks)
    })
  } catch {
    // Advisory only: never fail a request that already spent quota.
  }
}

/** Drop one task from the ledger once its file is on disk. */
async function forgetPendingTask(directory, taskId) {
  try {
    await withLedgerLock(directory, async () => {
      const tasks = await readPendingTasks(directory)
      if (tasks[taskId] === undefined) return
      delete tasks[taskId]
      await writePendingTasks(directory, tasks)
    })
  } catch {
    // Advisory only.
  }
}

/** Identify a supported video container from its signature bytes. */
function sniffVideoType(data) {
  const at = (offset, bytes) => bytes.every((byte, index) => data[offset + index] === byte)
  if (data.byteLength >= 12 && at(4, [0x66, 0x74, 0x79, 0x70])) return 'video/mp4'
  if (at(0, [0x1a, 0x45, 0xdf, 0xa3])) return 'video/webm'
  return undefined
}

/** File extension for one accepted video media type. */
function videoExtensionOf(mediaType) {
  return mediaType === 'video/webm' ? '.webm' : '.mp4'
}

/** Download one produced video into bytes plus its media type. */
async function downloadVideo(url, signal, maxBytes) {
  let response
  try {
    response = await fetch(url, { signal })
  } catch (error) {
    if (isAbortError(error)) throw error
    throw new QwenImageError(
      'downloading the generated video failed before a response: ' + (error?.message ?? String(error)),
      'QWEN_VIDEO_NETWORK',
    )
  }
  if (!response.ok) {
    throw new QwenImageError(
      'downloading the generated video failed with HTTP ' + response.status,
      'QWEN_VIDEO_DOWNLOAD_FAILED',
      response.status,
    )
  }
  const data = await readBodyLimited(response, maxBytes, 'the generated video', 'maxVideoBytes', 'QWEN_VIDEO_DOWNLOAD_TOO_LARGE')
  if (data.byteLength === 0) {
    throw new QwenImageError('the downloaded video was empty', 'QWEN_VIDEO_DOWNLOAD_FAILED')
  }
  const header = (response.headers.get('content-type') ?? '').split(';')[0]?.trim().toLowerCase() ?? ''
  const mediaType = header.startsWith('video/') ? header : sniffVideoType(data) ?? 'video/mp4'
  return { data, mediaType }
}

/** Sleep for a poll interval while honouring cancellation. */
function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason ?? new Error('aborted'))
      return
    }
    const onAbort = () => {
      clearTimeout(timer)
      reject(signal.reason ?? new Error('aborted'))
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

/** Combine the caller's cancellation with one request budget. */
function requestSignal(exec, budgetMs) {
  return combineSignals(exec.signal, AbortSignal.timeout(budgetMs))
}

/** Whether an error means "the caller stopped waiting", not "the request failed". */
function isAbortError(error) {
  return error?.name === 'AbortError' || error?.name === 'TimeoutError'
}

/** Retryable poll failures: dropped connections, rate limits, and provider 5xx. */
function isTransientQueryError(error) {
  if (error?.code === 'QWEN_VIDEO_NETWORK') return true
  if (error?.code !== 'QWEN_VIDEO_QUERY_FAILED') return false
  return error.status === 429 || (Number.isInteger(error.status) && error.status >= 500)
}

/** Wait before retrying a failed poll, honouring a Retry-After hint. */
function retryDelay(error, pollIntervalMs) {
  const hinted = Number(error?.retryAfterMs)
  if (Number.isFinite(hinted) && hinted > 0) return Math.min(hinted, MAX_RETRY_DELAY_MS)
  return Math.min(pollIntervalMs * 2, MAX_RETRY_DELAY_MS)
}

/** A locally raised cancellation, shaped like the platform's AbortError. */
function abortError(message = 'the tool call was cancelled') {
  const error = new Error(message)
  error.name = 'AbortError'
  return error
}

/** Whether a failed video download may be retried instead of surfaced. */
function isTransientVideoError(error) {
  if (error?.code === 'QWEN_VIDEO_NETWORK') return true
  if (error?.code !== 'QWEN_VIDEO_DOWNLOAD_FAILED') return false
  return error.status === 429 || (Number.isInteger(error.status) && error.status >= 500)
}

/**
 * Run one video download with bounded retries. The render is already paid for,
 * so a dropped transfer is retried here; the task id also stays in the ledger,
 * which is what makes a still-failing download recoverable by a later call.
 */
async function withVideoRetry(run, signal) {
  for (let attempt = 1; ; attempt += 1) {
    try {
      return await run()
    } catch (error) {
      if (isAbortError(error) || !isTransientVideoError(error) || attempt >= MAX_VIDEO_DOWNLOAD_ATTEMPTS) throw error
      await sleep(retryDelay(error, IMAGE_RETRY_DELAY_MS), signal)
    }
  }
}

/** Result returned when the render outlives this call's wait window. */
function pendingVideoResult(taskId, options, status, note) {
  return {
    model: options.model,
    taskId,
    status: 'pending',
    message: note === undefined
      ? 'The task is still ' + (status ?? 'RUNNING') + ' after ' + options.waitSeconds + ' seconds of polling.'
      : note,
  }
}

/** Validate and normalize the model-supplied video arguments. */
function parseVideoArgs(args, config) {
  if (args === null || typeof args !== 'object') throw new ImageArgsError('arguments must be an object')
  if (args.output_dir !== undefined && typeof args.output_dir !== 'string') {
    throw new ImageArgsError('output_dir must be a string')
  }
  const taskId = isNonBlankString(args.task_id) ? args.task_id.trim() : undefined
  const prompt = typeof args.prompt === 'string' ? args.prompt.trim() : ''
  const rawMedia = args.media
  let media
  if (rawMedia !== undefined) {
    if (!Array.isArray(rawMedia) || rawMedia.length === 0 || rawMedia.length > 8) {
      throw new ImageArgsError('media must be an array of 1 to 8 { type, url } entries')
    }
    media = rawMedia.map((entry) => {
      if (entry === null || typeof entry !== 'object') {
        throw new ImageArgsError('every media entry must be an object with type and url')
      }
      if (!VIDEO_MEDIA_TYPES.includes(entry.type)) {
        throw new ImageArgsError('media type "' + String(entry.type) + '" must be one of ' + VIDEO_MEDIA_TYPES.join(', '))
      }
      if (!isNonBlankString(entry.url)) throw new ImageArgsError('every media entry needs a non-empty url')
      return { type: entry.type, url: entry.url.trim() }
    })
  }
  if (taskId === undefined && prompt === '' && media === undefined) {
    throw new ImageArgsError('pass a prompt, at least one media entry, or a task_id to continue polling a submitted task')
  }

  const model = isNonBlankString(args.model) ? args.model.trim() : config.defaultVideoModel
  if (!KNOWN_VIDEO_MODELS.includes(model)) {
    throw new ImageArgsError('model "' + model + '" is not a Token Plan video model; use one of ' + KNOWN_VIDEO_MODELS.join(', '))
  }
  const resolution = isNonBlankString(args.resolution) ? args.resolution.trim().toUpperCase() : config.defaultVideoResolution
  if (!VIDEO_RESOLUTIONS.includes(resolution)) {
    throw new ImageArgsError('resolution "' + resolution + '" must be one of ' + VIDEO_RESOLUTIONS.join(', '))
  }
  const ratio = isNonBlankString(args.ratio) ? args.ratio.trim() : config.defaultVideoRatio
  if (!VIDEO_RATIOS.includes(ratio)) {
    throw new ImageArgsError('ratio "' + ratio + '" must be one of ' + VIDEO_RATIOS.join(', '))
  }
  const duration = args.duration === undefined ? config.defaultVideoDuration : args.duration
  if (!Number.isInteger(duration) || (duration !== -1 && (duration < 2 || duration > 30))) {
    throw new ImageArgsError('duration must be -1 (model decides) or an integer from 2 to 30 seconds')
  }
  const seed = args.seed
  if (seed !== undefined && (!Number.isInteger(seed) || seed < -1 || seed > 2_147_483_647)) {
    throw new ImageArgsError('seed must be -1 or an integer from 0 to 2147483647')
  }
  const waitSeconds = args.wait_seconds === undefined ? Math.floor(config.videoWaitMs / 1000) : args.wait_seconds
  if (!Number.isInteger(waitSeconds) || waitSeconds < 1 || waitSeconds > 1800) {
    throw new ImageArgsError('wait_seconds must be an integer from 1 to 1800')
  }
  return {
    taskId,
    prompt,
    media,
    model,
    resolution,
    ratio,
    ratioGiven: isNonBlankString(args.ratio),
    duration,
    seed,
    waitSeconds,
    audio: typeof args.audio === 'boolean' ? args.audio : undefined,
    watermark: typeof args.watermark === 'boolean' ? args.watermark : undefined,
    promptExtend: typeof args.prompt_extend === 'boolean' ? args.prompt_extend : undefined,
  }
}

/** Register the `qwen_image_generate` tool. */
export function apply(ctx, rawConfig) {
  const config = resolveConfig(rawConfig)

  ctx.tools.register({
    name: 'qwen_image_generate',
    description: 'Generate images from one prompt, or draw several different subjects in a single call '
      + '(wan2.7-image-pro by default, plus qwen-image-3.0-pro and others), save them to the session workspace, and return their paths. '
      + 'Use prompts (up to 6 entries) for a whole set, or count to redraw one prompt. name_prefix names the files, '
      + 'and every call appends a manifest recording the prompt and parameters. Pass image_urls to edit existing images. '
      + 'Generation takes tens of seconds per image.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        prompt: {
          type: 'string',
          description: 'What to draw, as a full image description. The service may truncate very long prompts. Pass prompts instead to draw several different images in one call.',
        },
        prompts: {
          type: 'array',
          items: { type: 'string' },
          description: 'Several different subjects in one call: 1 to 6 prompts, one image per prompt (count redraws each). One request per prompt; a failed prompt is reported in warnings while the others still return.',
        },
        name_prefix: {
          oneOf: [
            { type: 'string' },
            { type: 'array', items: { type: 'string' } },
          ],
          description: 'Base file name(s) for the saved images. A string prefixes every file (cat gives cat_01.png); an array names each image in order and must match the image count. Defaults to qwen_<timestamp>_<n>.',
        },
        model: {
          type: 'string',
          enum: KNOWN_MODELS,
          description: `Image model id. Defaults to "${config.defaultModel}".`,
        },
        size: {
          type: 'string',
          description: `Output resolution as width*height, such as 1024*1024, 1536*1024, or 720*1280. Defaults to "${config.defaultSize}".`,
        },
        count: {
          type: 'integer',
          description: 'How many images to produce per prompt, 1 through 6. Defaults to 1; prompts x count must stay within 6 images per call.',
        },
        negative_prompt: {
          type: 'string',
          description: 'What must not appear in the image. Only the qwen-image-3.0 series accepts this parameter; other models ignore it and the result reports a warning. At most 500 characters.',
        },
        image_urls: {
          type: 'array',
          items: { type: 'string' },
          description: 'Input images (public URLs or data URLs) to edit instead of generating from text alone; 1 through 3.',
        },
        seed: {
          type: 'integer',
          description: 'Random seed in [0, 2147483647] for reproducible output.',
        },
        watermark: {
          type: 'boolean',
          description: 'Add the "Qwen-Image" watermark. Defaults to the service setting.',
        },
        prompt_extend: {
          type: 'boolean',
          description: 'Let the service rewrite the prompt before drawing. Defaults to the service setting.',
        },
        output_dir: {
          type: 'string',
          description: 'Directory for the saved files; relative paths resolve against the session directory. Defaults to qwen-images under it.',
        },
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['model', 'images'],
        properties: {
          model: { type: 'string', description: 'Model that produced the images.' },
          images: {
            type: 'array',
            description: 'One entry per generated image, in request order.',
            items: {
              type: 'object',
              additionalProperties: false,
              required: ['path'],
              properties: {
                path: { type: 'string', description: 'Saved file path in the workspace.' },
                width: { type: 'integer', description: 'Pixel width reported by the service.' },
                height: { type: 'integer', description: 'Pixel height reported by the service.' },
                bytes: { type: 'integer', description: 'Saved byte size.' },
                mediaType: { type: 'string', description: 'Media type recorded for the image: the saved file\'s type, or the attachment store\'s normalized type when the image is shown in chat.' },
                attachmentId: { type: 'string', description: 'Durable attachment reference when the image is also shown in the chat.' },
                attachmentBytes: { type: 'integer', description: 'Normalized attachment byte size; the chat image block uses this, not the saved file size.' },
                promptIndex: { type: 'integer', description: 'Index of the prompt in prompts that produced this image (0 for a single prompt).' },
              },
            },
          },
          warnings: {
            type: 'array',
            items: { type: 'string' },
            description: 'Non-fatal notes: an ignored parameter, a failed prompt in a multi-prompt call, a rejected batch count, or a skipped download.',
          },
          manifestPath: { type: 'string', description: 'JSONL ledger recording every saved file with its prompt and parameters; appended in the output directory.' },
          rewriteStatus: { type: 'string', description: 'Prompt-rewrite status reported by the service, when present.' },
        },
      },
      render: (_args, value) => {
        const lines = value.images.map((image, index) => `${index + 1}. ${image.path}`
          + `${image.width === undefined || image.height === undefined ? '' : ` (${image.width}x${image.height}px)`}`)
        const blocks = [{
          type: 'text',
          text: `<generatedImages model="${value.model}">\n${lines.join('\n')}\n</generatedImages>`,
        }]
        for (const image of value.images) {
          if (image.attachmentId === undefined) continue
          blocks.push({
            type: 'image',
            attachment: {
              attachmentId: image.attachmentId,
              mediaType: image.mediaType ?? 'image/png',
              bytes: image.attachmentBytes ?? image.bytes ?? 0,
              width: image.width ?? 0,
              height: image.height ?? 0,
              name: image.path.split(/[\\/]/u).pop(),
            },
          })
        }
        if (Array.isArray(value.warnings) && value.warnings.length > 0) {
          blocks.push({
            type: 'text',
            text: '<warnings>' + String.fromCharCode(10) + value.warnings.map((warning) => '- ' + warning).join(String.fromCharCode(10)) + String.fromCharCode(10) + '</warnings>',
          })
        }
        return blocks
      },
      presentationMeta: (_args, value) => ({ paths: value.images.map((image) => image.path) }),
    },
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const options = parseArgs(args, config)
      const apiKey = await resolveApiKey(ctx, config.apiKeyEnv)
      const directory = await outputDirectory(ctx, config.outputDir, args.output_dir, exec, 'qwen-images')
      const attachments = ctx.get('attachments')
      // Only ask the model route about image input when an attachment could use it.
      const acceptsImages = attachments !== undefined && await routeAcceptsImages(ctx, exec)

      // One call budget, minus a download reserve: a finished image must still be
      // fetched after the budget expires, because its URL is short-lived and the
      // generation was already billed.
      const callDeadline = Date.now() + config.timeoutMs
      const reserve = Math.min(DOWNLOAD_RESERVE_MS, Math.floor(config.timeoutMs / 4))
      const generationTimeout = AbortSignal.timeout(Math.max(1_000, config.timeoutMs - reserve))
      const signal = combineSignals(exec.signal, generationTimeout)
      const request = {
        baseUrl: config.baseUrl,
        apiKey,
        model: options.model,
        size: options.size,
        negativePrompt: options.negativePrompt,
        seed: options.seed,
        watermark: options.watermark,
        promptExtend: options.promptExtend,
        images: options.images,
        signal,
      }
      const warnings = []

      // Negative prompts are a qwen-image-3.0-series parameter and the other
      // models reject unknown parameters, so the field is dropped there; report
      // that instead of letting the caller believe the constraint took effect.
      if (options.negativePrompt !== undefined && !options.model.startsWith('qwen-image-3.')) {
        warnings.push('negative_prompt was not sent: only the qwen-image-3.0 series accepts it, so ' + options.model + ' ignored it - move the constraint into prompt or switch models.')
      }
      if (options.negativePromptWasTrimmed) {
        warnings.push('negative_prompt was cut to its first 500 characters; shorten it so the whole constraint is sent.')
      }

      const prompts = options.prompts ?? [options.prompt]
      const requestOne = (prompt, count) => withImageRetry(
        () => requestGeneration({ ...request, prompt, count }),
        signal,
      )

      // One request per prompt. A model that rejects the batch count field is
      // retried as N single requests, and finished images are kept even when a
      // sibling request fails, so a paid result is never thrown away.
      const batches = await mapSettled(prompts, IMAGE_BATCH_CONCURRENCY, async (prompt, promptIndex) => {
        try {
          return { promptIndex, prompt, bodies: [await requestOne(prompt, options.count)] }
        } catch (error) {
          if (options.count <= 1 || !isBatchCountRejected(error)) throw error
          const settled = await Promise.allSettled(
            Array.from({ length: options.count }, () => requestOne(prompt, 1)),
          )
          const bodies = settled.filter((entry) => entry.status === 'fulfilled').map((entry) => entry.value)
          const failures = settled.filter((entry) => entry.status === 'rejected').map((entry) => entry.reason)
          if (bodies.length === 0) throw failures[0] ?? error
          warnings.push('the service rejected the batch count for prompt ' + (promptIndex + 1) + '; fell back to single requests (' + bodies.length + ' of ' + options.count + ' succeeded).')
          return { promptIndex, prompt, bodies }
        }
      })

      // A cancelled tool call is not a partial success: surface the cancellation
      // so the host reports it as cancelled instead of returning warnings.
      if (exec.signal?.aborted) {
        const cancelled = batches.find((entry) => entry.status === 'rejected')
        throw isAbortError(cancelled?.reason) ? cancelled.reason : abortError()
      }

      const produced = []
      let rewriteStatus
      for (let batchIndex = 0; batchIndex < batches.length; batchIndex += 1) {
        const outcome = batches[batchIndex]
        if (outcome.status === 'rejected') {
          const reason = outcome.reason
          warnings.push('prompt ' + (batchIndex + 1) + ' of ' + batches.length + ' failed: ' + (isAbortError(reason)
            ? 'the ' + Math.round(config.timeoutMs / 1000) + 's image budget ran out before the service answered; split the call or raise timeoutMs'
            : (reason?.message ?? String(reason))))
          continue
        }
        const { promptIndex, prompt, bodies } = outcome.value
        bodies.forEach((body, copyIndex) => {
          const entries = imageEntriesOf(body)
          if (rewriteStatus === undefined) rewriteStatus = entries.rewriteStatus
          const usage = body?.usage ?? {}
          const requestId = body?.request_id
          entries.images.forEach((source, imageIndex) => {
            produced.push({
              source,
              usage,
              requestId,
              prompt,
              promptIndex,
              slot: promptIndex * options.count + copyIndex + imageIndex,
            })
          })
        })
      }
      if (produced.length === 0) {
        const firstFailure = batches.find((entry) => entry.status === 'rejected')
        if (firstFailure !== undefined) {
          const reason = firstFailure.reason
          // A budget expiry deserves its own code: "split the batch" is
          // actionable, a bare timeout exception is not.
          if (generationTimeout.aborted && isAbortError(reason) && !exec.signal?.aborted) {
            throw new QwenImageError(
              'the ' + Math.round(config.timeoutMs / 1000) + 's image budget ran out before any image arrived; pass fewer prompts per call, retry later, or raise timeoutMs',
              'QWEN_IMAGE_TIMEOUT',
            )
          }
          throw reason instanceof Error ? reason : new QwenImageError(String(reason), 'QWEN_IMAGE_REQUEST_FAILED')
        }
        throw new QwenImageError('the service returned no image; retry with a different prompt or model', 'QWEN_IMAGE_NO_IMAGE')
      }

      await mkdir(directory, { recursive: true })
      const batch = stamp()
      // Generation may have consumed its whole budget; the download phase gets
      // its own short window so already paid images still reach the disk.
      const downloadBudget = Math.min(DOWNLOAD_CEILING_MS, Math.max(DOWNLOAD_FLOOR_MS, callDeadline - Date.now()))
      const downloadSignal = combineSignals(exec.signal, AbortSignal.timeout(downloadBudget))
      // Downloading is the slow half and the images are independent, so every
      // file is fetched at once; naming and writing stay sequential because the
      // exclusive-create name check is what keeps concurrent calls apart.
      const downloads = await Promise.all(produced.map(async (item) => {
        try {
          const downloaded = await withImageRetry(
            () => downloadImage(item.source, downloadSignal, config.maxImageBytes),
            downloadSignal,
          )
          return { ...item, ...downloaded }
        } catch (error) {
          return { ...item, error }
        }
      }))
      // A cancelled call must not finish as a failed download; report it as the
      // cancellation it is.
      if (exec.signal?.aborted) throw abortError()

      const saved = []
      const manifest = []
      for (const item of downloads) {
        if (item.error !== undefined) {
          warnings.push('image from ' + shortSource(item.source) + ' could not be downloaded: ' + (isAbortError(item.error)
            ? 'the download window closed before the file arrived; the service URL may already have expired'
            : (item.error?.message ?? String(item.error))))
          continue
        }
        const fallbackStem = 'qwen_' + batch + '_' + (item.slot + 1)
        const stem = options.namePrefixes === undefined
          ? fallbackStem
          : options.namePrefixIsList
            ? (options.namePrefixes[item.slot] ?? fallbackStem)
            : options.namePrefixes[0] + '_' + String(item.slot + 1).padStart(2, '0')
        const file = await writeUniqueFile(directory, stem, extensionOf(item.mediaType), item.data)
        const entry = { path: file, bytes: item.data.byteLength, mediaType: item.mediaType, promptIndex: item.promptIndex }
        if (acceptsImages && attachments !== undefined) {
          try {
            const ref = await attachments.saveImage({ data: item.data, mediaType: item.mediaType, name: baseNameOf(file) })
            entry.attachmentId = ref.attachmentId
            // The attachment service normalizes the bytes it keeps, so the
            // durable reference carries the stored size that the request path
            // verifies against; the saved file keeps the downloaded size.
            entry.attachmentBytes = ref.bytes
            entry.width = ref.width
            entry.height = ref.height
            // Only replace the recorded type when the store reports one; the
            // saved file keeps the extension of the type it was downloaded as.
            if (isNonBlankString(ref.mediaType)) entry.mediaType = ref.mediaType
          } catch {
            // A refused attachment still leaves the saved file for the model to read.
          }
        }
        if (entry.width === undefined) {
          const width = item.usage.output_width ?? item.usage.width
          const height = item.usage.output_height ?? item.usage.height
          if (Number.isInteger(width) && Number.isInteger(height)) {
            entry.width = width
            entry.height = height
          }
        }
        saved.push(entry)
        manifest.push({
          createdAt: new Date().toISOString(),
          kind: 'image',
          file,
          prompt: item.prompt,
          promptIndex: item.promptIndex,
          model: options.model,
          size: options.size,
          seed: options.seed,
          negativePrompt: options.negativePrompt,
          promptExtend: options.promptExtend,
          watermark: options.watermark,
          requestId: item.requestId,
          width: entry.width,
          height: entry.height,
          bytes: entry.bytes,
          mediaType: entry.mediaType,
        })
      }
      if (saved.length === 0) {
        const failed = downloads.find((item) => item.error !== undefined)
        throw new QwenImageError(
          'no image could be saved: ' + (failed?.error?.message ?? 'every download failed')
            + (failed?.source === undefined ? '' : ' (source: ' + shortSource(failed.source) + ')'),
          'QWEN_IMAGE_DOWNLOAD_FAILED',
        )
      }

      let manifestPath = join(directory, MANIFEST_FILE)
      try {
        manifestPath = await appendManifest(directory, manifest)
      } catch (error) {
        warnings.push('the generation manifest could not be written: ' + (error?.message ?? String(error)))
      }
      return {
        model: options.model,
        images: saved,
        manifestPath,
        ...warnings.length === 0 ? {} : { warnings },
        ...rewriteStatus === undefined ? {} : { rewriteStatus },
      }
    },
    presentCall(args) {
      const input = args !== null && typeof args === 'object' ? args : {}
      return {
        card: 'generic',
        title: `Generate image (${typeof input.model === 'string' ? input.model : config.defaultModel})`,
        kind: 'other',
        rawInput: {
          prompt: typeof input.prompt === 'string' ? input.prompt : Array.isArray(input.prompts) ? input.prompts.join(' / ') : undefined,
          size: typeof input.size === 'string' ? input.size : config.defaultSize,
        },
      }
    },
  })

  ctx.tools.register({
    name: 'qwen_video_generate',
    description: 'Generate a short video from a text prompt (and optionally first-frame, last-frame, or reference media) '
      + 'with the Qwen Token Plan video models (wan3.0-video by default). Submission is asynchronous: this call polls for '
      + 'up to wait_seconds (default ' + Math.floor(config.videoWaitMs / 1000) + 's, chosen to fit inside the host default '
      + 'tool timeout) and returns status "pending" with a task_id when the render is still running — call it again with '
      + 'that task_id to keep waiting and download the finished MP4. A pending result has no video field, only taskId and '
      + 'message; submitted ids are also listed in .pending-tasks.json in the output directory. Rendering usually takes '
      + 'between one and several minutes.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        prompt: {
          type: 'string',
          description: 'What should happen in the video, as a full shot description. Required unless media or task_id is given.',
        },
        model: {
          type: 'string',
          enum: KNOWN_VIDEO_MODELS,
          description: 'Video model id. wan3.0-video handles text, first/last frame, and all-reference input; '
            + 'happyhorse-1.1-t2v/i2v/r2v are text, first-frame, and reference specializations. Defaults to "' + config.defaultVideoModel + '".',
        },
        resolution: {
          type: 'string',
          enum: VIDEO_RESOLUTIONS,
          description: 'Resolution tier. Defaults to "' + config.defaultVideoResolution + '".',
        },
        ratio: {
          type: 'string',
          enum: VIDEO_RATIOS,
          description: 'Aspect ratio; "adaptive" lets wan3.0-video follow the input media. Defaults to "' + config.defaultVideoRatio + '".',
        },
        duration: {
          type: 'integer',
          description: 'Length in seconds: -1 (model decides) or 2 through 30. Defaults to ' + config.defaultVideoDuration + '.',
        },
        audio: {
          type: 'boolean',
          description: 'Whether the output carries an audio track, for models that support sound. Defaults to the service setting.',
        },
        watermark: {
          type: 'boolean',
          description: 'Add the model watermark. Defaults to the service setting.',
        },
        seed: {
          type: 'integer',
          description: 'Random seed: -1 or 0 through 2147483647.',
        },
        prompt_extend: {
          type: 'boolean',
          description: 'Let the service rewrite the prompt before rendering. Defaults to the service setting.',
        },
        media: {
          type: 'array',
          description: 'Input media slots. first_frame/last_frame cannot be mixed with reference_* or file/link; '
            + 'reference_image/reference_video/reference_audio combine freely. Order defines "image 1"/"video 1" in the prompt.',
          items: {
            type: 'object',
            additionalProperties: false,
            required: ['type', 'url'],
            properties: {
              type: { type: 'string', enum: VIDEO_MEDIA_TYPES },
              url: { type: 'string', description: 'Public HTTP(S) URL, oss:// temporary URL, or data:{MIME};base64,... payload.' },
            },
          },
        },
        task_id: {
          type: 'string',
          description: 'Continue waiting for an already submitted task instead of submitting a new one.',
        },
        wait_seconds: {
          type: 'integer',
          description: 'How long this call keeps polling before returning a pending task id. Defaults to '
            + Math.floor(config.videoWaitMs / 1000) + ' seconds.',
        },
        output_dir: {
          type: 'string',
          description: 'Directory for the saved file; relative paths resolve against the session directory. Defaults to qwen-videos under it.',
        },
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['model', 'taskId', 'status'],
        properties: {
          model: { type: 'string', description: 'Model that produced (or is producing) the video.' },
          taskId: { type: 'string', description: 'Task id; pass it back to keep polling a pending task.' },
          status: { type: 'string', enum: ['succeeded', 'pending'], description: 'Whether the MP4 is downloaded or still rendering.' },
          video: {
            type: 'object',
            additionalProperties: false,
            required: ['path'],
            properties: {
              path: { type: 'string', description: 'Saved MP4 path in the workspace.' },
              bytes: { type: 'integer', description: 'Saved byte size.' },
              mediaType: { type: 'string', description: 'Saved video media type.' },
              durationSeconds: { type: 'integer', description: 'Video length reported by the service, in seconds.' },
              ratio: { type: 'string', description: 'Aspect ratio reported by the service.' },
            },
          },
          message: { type: 'string', description: 'Why the call returned without a file, when it did.' },
          warnings: {
            type: 'array',
            items: { type: 'string' },
            description: 'Non-fatal notes: arguments ignored while resuming a task, or a finished render whose download still needs the task id.',
          },
        },
      },
      render: (_args, value) => {
        const blocks = []
        if (value.status === 'pending') {
          blocks.push({
            type: 'text',
            text: '<videoTask model="' + value.model + '" taskId="' + value.taskId + '" status="pending">\n'
              + (value.message ?? 'The video is still rendering.') + '\n'
              + 'Call qwen_video_generate again with task_id "' + value.taskId + '" to keep waiting and download the finished MP4.\n'
              + '</videoTask>',
          })
        } else {
          const video = value.video
          const details = [
            video.bytes === undefined ? '' : video.bytes + ' bytes',
            video.mediaType ?? '',
            video.durationSeconds === undefined ? '' : video.durationSeconds + 's',
            video.ratio ?? '',
          ].filter((entry) => entry !== '').join(', ')
          blocks.push({
            type: 'text',
            text: '<generatedVideo model="' + value.model + '" taskId="' + value.taskId + '">\n'
              + '1. ' + video.path + (details === '' ? '' : ' (' + details + ')') + '\n'
              + '</generatedVideo>',
          })
        }
        if (Array.isArray(value.warnings) && value.warnings.length > 0) {
          blocks.push({
            type: 'text',
            text: '<warnings>' + String.fromCharCode(10) + value.warnings.map((warning) => '- ' + warning).join(String.fromCharCode(10)) + String.fromCharCode(10) + '</warnings>',
          })
        }
        return blocks
      },
      presentationMeta: (_args, value) => (value.video === undefined
        ? { taskId: value.taskId }
        : { paths: [value.video.path] }),
    },
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const options = parseVideoArgs(args, config)
      const apiKey = await resolveApiKey(ctx, config.apiKeyEnv)
      const directory = await outputDirectory(ctx, config.outputDir, args.output_dir, exec, 'qwen-videos')
      // Validate the endpoint before submitting: a malformed videoBaseUrl must
      // fail before a billed task exists, not after it.
      videoTaskUrl(config.videoBaseUrl, 'probe')

      const warnings = []
      if (options.waitSeconds * 1000 > HOST_DEFAULT_TIMEOUT_MS) {
        warnings.push('wait_seconds is longer than the host default per-call timeout (' + Math.round(HOST_DEFAULT_TIMEOUT_MS / 1000) + 's); if the host kills this call, resume with the task_id it returned instead of resubmitting.')
      }
      const ledger = await readPendingTasks(directory)
      const recorded = options.taskId === undefined ? undefined : ledger[options.taskId]
      if (recorded !== undefined && isNonBlankString(recorded.model) && KNOWN_VIDEO_MODELS.includes(recorded.model)) {
        // The ledger knows what was submitted: resuming by task id must not
        // restate the model, and the result must not misreport it either.
        if (isNonBlankString(args.model) && args.model.trim() !== recorded.model) {
          warnings.push('model "' + args.model.trim() + '" was ignored on resume: task ' + options.taskId + ' was submitted with ' + recorded.model + '.')
        }
        options.model = recorded.model
      }
      if (options.taskId !== undefined && (options.prompt !== '' || options.media !== undefined)) {
        warnings.push('task_id was given, so prompt/media were ignored; the already submitted task is polled as-is.')
      }

      let taskId = options.taskId
      let submission = recorded
      if (taskId === undefined) {
        const input = {}
        if (options.prompt !== '') input.prompt = options.prompt
        if (options.media !== undefined) input.media = options.media
        const parameters = {
          resolution: options.resolution,
          duration: options.duration,
        }
        // Wan 3.0 owns adaptive ratios and audio tracks; HappyHorse derives the
        // ratio from its first frame and rejects these parameters. A first/last
        // frame request keeps the frame's own shape unless a ratio was asked for.
        const followsFrame = options.media !== undefined
          && options.media.some((entry) => entry.type === 'first_frame' || entry.type === 'last_frame')
        if (options.ratioGiven || options.model.startsWith('wan3.0')) {
          parameters.ratio = followsFrame && !options.ratioGiven ? 'adaptive' : options.ratio
        }
        if (options.audio !== undefined && options.model.startsWith('wan3.0')) parameters.audio = options.audio
        if (options.seed !== undefined) parameters.seed = options.seed
        if (options.watermark !== undefined) parameters.watermark = options.watermark
        // The service requires prompt rewriting when a document or web page is
        // attached, so a file/link slot forces it on unless already requested.
        const needsExtend = options.media !== undefined
          && options.media.some((entry) => entry.type === 'file' || entry.type === 'link')
        if (options.promptExtend !== undefined) parameters.prompt_extend = options.promptExtend
        else if (needsExtend) parameters.prompt_extend = true

        try {
          taskId = await submitVideoTask({
            baseUrl: config.videoBaseUrl,
            apiKey,
            model: options.model,
            input,
            parameters,
            // A create call is quick: a budget under the host's kill window keeps
            // its failure mode a clean timeout instead of a killed process.
            signal: requestSignal(exec, Math.min(config.videoTimeoutMs, VIDEO_SUBMIT_BUDGET_MS)),
          })
        } catch (error) {
          if (isAbortError(error) && !exec.signal?.aborted) {
            throw new QwenImageError(
              'the video submission timed out before a task id came back, so a task may exist and be billing; check the Token Plan console before resubmitting, because a resubmission is charged again',
              'QWEN_VIDEO_SUBMIT_UNKNOWN',
            )
          }
          throw error
        }
        submission = {
          model: options.model,
          prompt: options.prompt === '' ? undefined : options.prompt,
          parameters,
          submittedAt: new Date().toISOString(),
        }
      }

      const taskUrl = videoTaskUrl(config.videoBaseUrl, taskId)
      const deadline = Date.now() + options.waitSeconds * 1000
      await mkdir(directory, { recursive: true })
      // The task is billed the moment it is submitted, so its id is recorded
      // before the first poll: a killed process or a cancelled wait can resume
      // the render instead of paying for a second one.
      if (options.taskId === undefined || recorded === undefined) {
        await rememberPendingTask(directory, taskId, {
          model: options.model,
          directory,
          submittedAt: submission?.submittedAt ?? new Date().toISOString(),
          prompt: options.prompt === '' ? undefined : options.prompt,
          parameters: submission?.parameters,
        })
      }
      const result = (value) => (warnings.length === 0 ? value : { ...value, warnings })
      const pending = (taskStatus, note) => result(pendingVideoResult(taskId, options, taskStatus, note))

      let body
      let status
      let unknownStreak = 0
      let failureStreak = 0
      for (;;) {
        // One query must never outlive the caller's own wait budget, or a hung
        // request would overshoot wait_seconds by the whole request budget.
        const remaining = deadline - Date.now()
        if (remaining <= 0) {
          return pending(status)
        }
        try {
          body = await queryVideoTask(
            taskUrl,
            apiKey,
            requestSignal(exec, Math.min(config.videoTimeoutMs, Math.max(1_000, remaining))),
          )
          failureStreak = 0
        } catch (error) {
          if (isAbortError(error)) {
            return pending(status, 'the wait was cancelled while the task was still rendering; the task id stays valid for 24 hours')
          }
          if (!isTransientQueryError(error) || failureStreak >= MAX_QUERY_FAILURES) throw error
          failureStreak += 1
          try {
            await sleep(retryDelay(error, config.videoPollIntervalMs), exec.signal)
          } catch (sleepError) {
            if (isAbortError(sleepError)) {
              return pending(status, 'the wait was cancelled while the task was still rendering')
            }
            throw sleepError
          }
          continue
        }
        status = videoTaskStatus(body)
        if (status === 'SUCCEEDED') break
        if (status === 'FAILED' || status === 'CANCELED') {
          await forgetPendingTask(directory, taskId)
          throw new QwenImageError(
            'video task ' + taskId + ' ended as ' + status + ': ' + videoTaskFailure(body),
            'QWEN_VIDEO_TASK_FAILED',
          )
        }
        if (status === 'UNKNOWN') {
          // A status-less answer is usually one malformed poll, not a dead render.
          // A long run of them means polling cannot make progress, but the task
          // may still be rendering and billing: keep its id in the ledger and let
          // the caller decide instead of dropping the only proof of the charge.
          unknownStreak += 1
          if (unknownStreak > MAX_UNKNOWN_POLLS) {
            return pending(status, 'the service kept answering without a task status; the task id stays valid for 24 hours, so retry with task_id "' + taskId + '" before resubmitting')
          }
        } else {
          unknownStreak = 0
        }
        if (Date.now() + config.videoPollIntervalMs > deadline) {
          return pending(status)
        }
        try {
          await sleep(config.videoPollIntervalMs, exec.signal)
        } catch (error) {
          if (isAbortError(error)) {
            return pending(status, 'the wait was cancelled while the task was still rendering')
          }
          throw error
        }
      }

      const source = videoSourceOf(body)
      if (source === undefined) {
        await forgetPendingTask(directory, taskId)
        throw new QwenImageError('the finished video task carried no video URL', 'QWEN_VIDEO_NO_VIDEO')
      }
      let downloaded
      try {
        downloaded = await withVideoRetry(
          () => downloadVideo(source, requestSignal(exec, Math.min(config.videoTimeoutMs, VIDEO_DOWNLOAD_BUDGET_MS)), config.maxVideoBytes),
          exec.signal,
        )
      } catch (error) {
        if (isAbortError(error)) {
          // The render finished and was billed; the ledger keeps its id so a
          // later call can fetch the file without paying for a second render.
          return result(pendingVideoResult(
            taskId,
            options,
            'SUCCEEDED',
            'the video finished rendering but the download was interrupted; call qwen_video_generate again with task_id "' + taskId + '" to fetch it (no new render, no new charge)',
          ))
        }
        throw error
      }
      const { data, mediaType } = downloaded
      const file = await writeUniqueFile(directory, 'video_' + stamp() + '_1', videoExtensionOf(mediaType), data)
      await forgetPendingTask(directory, taskId)
      const video = { path: file, bytes: data.byteLength, mediaType }
      const usage = body?.usage ?? {}
      const durationSeconds = usage.video_duration ?? usage.duration
      if (Number.isInteger(durationSeconds)) video.durationSeconds = durationSeconds
      const ratio = usage.video_ratio ?? usage.ratio
      if (isNonBlankString(ratio)) video.ratio = ratio
      const parameters = submission?.parameters ?? {}
      const record = {
        createdAt: new Date().toISOString(),
        kind: 'video',
        file,
        taskId,
        model: options.model,
        prompt: submission?.prompt,
        resolution: parameters.resolution,
        duration: parameters.duration,
        ratio: parameters.ratio,
        seed: parameters.seed,
        audio: parameters.audio,
        watermark: parameters.watermark,
        promptExtend: parameters.prompt_extend,
        requestId: body?.request_id,
        bytes: video.bytes,
        mediaType: video.mediaType,
        durationSeconds: video.durationSeconds,
        renderedRatio: video.ratio,
      }
      try {
        await appendManifest(directory, [record])
      } catch (error) {
        warnings.push('the generation manifest could not be written: ' + (error?.message ?? String(error)))
      }
      return result({ model: options.model, taskId, status: 'succeeded', video })
    },
    presentCall(args) {
      const input = args !== null && typeof args === 'object' ? args : {}
      return {
        card: 'generic',
        title: 'Generate video (' + (typeof input.model === 'string' ? input.model : config.defaultVideoModel) + ')',
        kind: 'other',
        rawInput: {
          prompt: typeof input.prompt === 'string' ? input.prompt : undefined,
          duration: typeof input.duration === 'number' ? input.duration : config.defaultVideoDuration,
          resolution: typeof input.resolution === 'string' ? input.resolution : config.defaultVideoResolution,
        },
      }
    },
  })
}
