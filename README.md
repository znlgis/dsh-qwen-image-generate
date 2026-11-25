# 千问生图与视频（@dsh-gis/dsh-qwen-image-generate）

给 DSH 增加两个工具：用千问 Token Plan 的模型按文字描述生成图片或视频，产物下载到当前会话的工作目录。

- `qwen_image_generate`：同步调用图像生成接口，图片直接显示在对话里。
- `qwen_video_generate`：提交异步视频任务并轮询，成片下载为 MP4（渲染通常要一到几分钟）。

## 用法

在对话里直接说「用千问画一张……」「生成一段……的视频」，模型会调用对应工具。也可以显式指定参数。

### 图片

- 模型：`wan2.7-image-pro`（默认）、`wan2.7-image`、`qwen-image-3.0-pro`、`qwen-image-2.0-pro`、`qwen-image-2.0`
- 尺寸：`宽*高`，例如 `1024*1024`（默认）、`1536*1024`、`720*1280`
- 一次最多 6 张：`count` 重画同一提示词，`prompts` 一次画一组不同主体（1~6 条，每条一张图，`count` 对每条生效，
  `prompts.length × count` 不能超过 6）。系列图用 `prompts` 一次出完，不要用多次调用拼
- `name_prefix` 指定文件名：字符串给所有文件加前缀（`cat` → `cat_01.png`），数组按顺序逐个命名且数量必须与图片数一致；
  默认仍是 `qwen_<时间戳>_<序号>.png`
- 服务不支持批量字段（`n`）时自动退回逐张请求，并保留已成功的图片：失败的那张写进 `warnings`，整批不会因为一张失败被丢弃
- `negative_prompt` 只对 `qwen-image-3.0` 系列生效；其他模型不会发送该参数，结果里会给出 `warnings` 提示
- 每次调用会把提示词、模型、尺寸、seed 等写成一行追加到输出目录的 `.manifest.jsonl`，返回的 `manifestPath` 指向它；视频成片同样记录（`kind: "video"`），重复出图/复现时先看这里
- 传入 `image_urls` 可让支持图生图的模型改图

### 视频

- 模型：`wan3.0-video`（默认，支持文生视频、首帧/首尾帧、全能参考、有声视频）、`wan3.0-video-prime`；
  也可用 `happyhorse-1.1-t2v`、`happyhorse-1.1-i2v`、`happyhorse-1.1-r2v`
- `resolution`：`480P`、`720P`（默认）、`1080P`
- `ratio`：`16:9`（默认）、`9:16`、`1:1`、`4:3`、`3:4`、`21:9`、`adaptive` 等；
  传首帧/尾帧且没显式指定比例时自动用 `adaptive`，保住原图构图
- `duration`：秒数，`-1` 交给模型决定，或 2~30（默认 5）
- `media`：输入素材数组，每项 `{ "type": "...", "url": "..." }`，类型见下：
  `first_frame`、`last_frame`、`reference_image`、`reference_video`、`reference_audio`、`file`、`link`。
  `first_frame`/`last_frame` 与参考类素材互斥；`file`/`link` 可与参考类组合。素材顺序决定提示词里「图1」「视频1」的指代。
  传 `file`/`link` 时服务要求开启提示词改写，插件会自动带上 `prompt_extend: true`。
- 其余可选参数：`audio`（是否带音轨）、`watermark`、`seed`、`prompt_extend`
- 结果里的 `warnings` 会说明被忽略的参数（例如用 `task_id` 续查时又传了 prompt），以及渲染完成但下载仍需续取的情况

视频是异步任务：这次调用最多等 `wait_seconds`（默认 90 秒），还没渲染完就返回 `status: "pending"` 和
`task_id`；再用 `task_id` 调一次即可继续等待并下载。任务 id 的有效期是 24 小时。

默认 90 秒是刻意压在宿主默认的单次工具超时（120 秒）之下：等待窗口一旦超过宿主超时，进程会在轮询途中被杀，
任务已提交并计费，`task_id` 却跟着丢失。需要更长的单次等待就显式传 `wait_seconds`（上限 1800），并把宿主的
工具超时一起调高；超过 120 秒时结果会附 `warnings` 提醒这一点。轮询途中被取消、或服务连续返回不出任务状态时，
都会返回 `pending` 而不是抛错，已提交的任务记在输出目录的 `.pending-tasks.json`，进程被杀后照着里面的
`taskId` 继续查即可。成片已渲染完但下载被中断时同样返回
`pending` 并保留 `taskId`，再调一次只下载文件，不会重新渲染、也不重复计费。
续查时请传与提交时相同的 `output_dir`：待办账本 `.pending-tasks.json` 就放在那个目录里。

生成的文件保存在会话目录下的 `qwen-images/` 与 `qwen-videos/`，命名形如
`qwen_20261125_140901_1.png`、`video_20261125_140901_1.mp4`。
可用 `output_dir` 参数改到别的目录（相对路径按会话目录解析，且必须留在会话工作区内）。

## 配置 API Key

插件不保存密钥本身，只保存**凭据名**。默认读取 `DASHSCOPE_API_KEY`（与本仓库 `cordis.patch.yml` 的取值一致）：

1. 到千问 Token Plan 控制台获取以 `sk-sp-` 开头的 API Key（[接入概览](https://platform.qianwenai.com/docs/token-plan)）。
2. 在 DSH 的模型与凭据设置里保存为 `DASHSCOPE_API_KEY`；或在启动 DSH 前 `export DASHSCOPE_API_KEY=sk-sp-...`。

凭据解析顺序由 DSH 凭据服务决定：存储的凭据 > 启动环境变量 > 项目 `.env` > harness home 的 `.env`。

## 可配置项

编辑插件行（**插件** → 该插件 → 配置）即可修改，对应 `cordis.patch.yml` 中该行的 `config`：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `apiKeyEnv` | `DASHSCOPE_API_KEY`（`cordis.patch.yml` 中的取值） | 凭据名，不是密钥值 |
| `baseUrl` | `.../services/aigc/multimodal-generation/generation` | 图像生成接口 |
| `defaultModel` | `wan2.7-image-pro` | 生图未指定模型时使用 |
| `defaultSize` | `1024*1024` | 生图未指定尺寸时使用 |
| `timeoutMs` | `110000` | 单次生图调用的总预算（预留约 15 秒给下载已完成的图片），刻意压在宿主 120 秒工具超时之下 |
| `videoBaseUrl` | `.../services/aigc/video-generation/video-synthesis` | 视频提交接口；任务查询地址由它派生 |
| `defaultVideoModel` | `wan3.0-video` | 视频未指定模型时使用 |
| `defaultVideoResolution` | `720P` | 视频未指定分辨率时使用 |
| `defaultVideoRatio` | `16:9` | 视频未指定比例时使用 |
| `defaultVideoDuration` | `5` | 视频未指定时长（秒）时使用 |
| `videoTimeoutMs` | `180000` | 单次提交/查询请求的预算；成片下载另行封顶 60 秒，失败可凭 task_id 续取 |
| `videoPollIntervalMs` | `5000` | 轮询间隔 |
| `videoWaitMs` | `90000` | 单次工具调用默认等待时长，超时返回 task_id；刻意压在宿主默认 120 秒超时之下 |
| `outputDir` | 空（用会话目录） | 会话目录不可用时的兜底目录 |
| `maxImageBytes` | `67108864` | 单张图片下载上限（64 MiB），超限即中断并提示调高 |
| `maxVideoBytes` | `536870912` | 单个视频下载上限（512 MiB），超限即中断并提示调高 |

## 说明与限制

- 生成会把提示词（以及图生图/参考生视频时的输入素材）发送到千问 Token Plan 接口，按套餐用量计费；视频消耗高于图片。
- `output_dir` 只能指向会话工作区内，越界路径会被拒绝；判断时会解析已有的符号链接/junction，工作区里指向外部的链接也绕不过去。配置项 `outputDir` 是运维配置，不受这条限制。
- 单次生图调用默认预算 `timeoutMs` 为 110 秒，刻意压在宿主默认 120 秒的每次工具调用超时之下：生图是同步接口，没有 task id 可续查，
  预算一旦超过宿主限制，过程会被宿主杀掉而图已计费。预算里预留约 15 秒给下载：生成阶段超时后仍会保存已经生成并计费的图片，
  一张都没生成时才返回 `QWEN_IMAGE_TIMEOUT` 并提示拆分调用。批量时按单张十几到几十秒估算，宁可分两次调用。
- 只有服务明确拒绝批量字段 `n` 时才会退回逐张请求；尺寸、种子等其他 400 不会被按图片数重复请求，避免为一个不可能成功的请求重复计费。
- 已提交但没等到结果的视频任务记在输出目录的 `.pending-tasks.json`，任务完成或明确失败后自动移除；
  账本按输出目录串行更新，并且先写临时文件再改名覆盖，进程中途被杀不会留下半截账本；解析失败的账本会改名成
  `.pending-tasks.json.corrupt-<时间戳>` 保留现场，而不是被下一次写入静默清空。
- 视频提交若超时，插件无法确认任务是否已创建，会返回 `QWEN_VIDEO_SUBMIT_UNKNOWN` 并提示先看 Token Plan 控制台；
  直接重提会重复计费。
- 图片/视频下载都有字节上限（`maxImageBytes` / `maxVideoBytes`），流式读取超限即中断，异常响应不会撑爆内存。
- 接口返回的图片/视频链接有时效，插件会立即下载到本地，因此不依赖该链接。
- 图片只有在当前模型声明支持图片输入时才作为图片块进入对话；纯文本模型只会收到文件路径。视频只返回本地路径。
- 账本的并发保护只在单个 DSH 进程内生效：多个进程同时往同一输出目录提交视频时，仍可能互相覆盖（改名替换保证不会留下半截文件）。
  共享同一工作目录的多实例场景请避免同时提交视频，或给每个实例配不同的 `output_dir`。
- 关闭插件后工具立即消失，已下载的文件保留在工作目录中。

## 开发与测试

```bash
npm test        # 离线测试：桩掉 fetch 与 ctx 服务，不联网、不消耗额度
npm run check   # 两个入口文件的语法检查
```

测试覆盖参数校验、批量回退、重试预算、输出目录越界、并发命名、pending 任务账本、超时与下载中断等路径；
CI 在 Node 20/22/24 上跑同一套测试，并用 `npm pack --dry-run` 校验发布内容。

## 接口参考

- [接入图像生成模型](https://platform.qianwenai.com/docs/token-plan/best-practices/image-generation)
- [接入视频生成模型](https://platform.qianwenai.com/docs/token-plan/best-practices/video-generation)
- [wan3.0-video 视频生成 API](https://platform.qianwenai.com/docs/api-reference/video-generation)
- [Qwen 图像生成与编辑 API](https://platform.qianwenai.com/docs/api-reference/image-generation/qwen-image-editing)
