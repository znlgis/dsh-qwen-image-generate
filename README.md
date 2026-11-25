# 千问生图与视频（@dsh-gis/dsh-qwen-image-generate）

给 DSH 增加两个工具：用千问 Token Plan 的模型按文字描述生成图片或视频，产物下载到当前会话的工作目录。

- `qwen_image_generate`：同步调用图像生成接口，图片直接显示在对话里。
- `qwen_video_generate`：提交异步视频任务并轮询，成片下载为 MP4（渲染通常要一到几分钟）。

## 用法

在对话里直接说「用千问画一张……」「生成一段……的视频」，模型会调用对应工具。也可以显式指定参数。

### 图片

- 模型：`wan2.7-image-pro`（默认）、`wan2.7-image`、`qwen-image-3.0-pro`、`qwen-image-2.0-pro`、`qwen-image-2.0`
- 尺寸：`宽*高`，例如 `1024*1024`（默认）、`1536*1024`、`720*1280`
- 一次最多 6 张（`count`）；传入 `image_urls` 可让支持图生图的模型改图

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

视频是异步任务：这次调用最多等 `wait_seconds`（默认 300 秒），还没渲染完就返回 `status: "pending"` 和
`task_id`；再用 `task_id` 调一次即可继续等待并下载。任务 id 的有效期是 24 小时。

生成的文件保存在会话目录下的 `qwen-images/` 与 `qwen-videos/`，命名形如
`qwen_20261125_140901_1.png`、`video_20261125_140901_1.mp4`。
可用 `output_dir` 参数改到别的目录（相对路径按会话目录解析）。

## 配置 API Key

插件不保存密钥本身，只保存**凭据名**。默认读取 `QWEN_TOKEN_PLAN_API_KEY`：

1. 到千问 Token Plan 控制台获取以 `sk-sp-` 开头的 API Key（[接入概览](https://platform.qianwenai.com/docs/token-plan)）。
2. 在 DSH 的模型与凭据设置里保存为 `QWEN_TOKEN_PLAN_API_KEY`；或在启动 DSH 前 `export QWEN_TOKEN_PLAN_API_KEY=sk-sp-...`。

凭据解析顺序由 DSH 凭据服务决定：存储的凭据 > 启动环境变量 > 项目 `.env` > harness home 的 `.env`。

## 可配置项

编辑插件行（**插件** → 该插件 → 配置）即可修改，对应 `cordis.patch.yml` 中该行的 `config`：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `apiKeyEnv` | `DASHSCOPE_API_KEY`（`cordis.patch.yml` 中的取值） | 凭据名，不是密钥值 |
| `baseUrl` | `.../services/aigc/multimodal-generation/generation` | 图像生成接口 |
| `defaultModel` | `wan2.7-image-pro` | 生图未指定模型时使用 |
| `defaultSize` | `1024*1024` | 生图未指定尺寸时使用 |
| `timeoutMs` | `180000` | 单次生图请求与下载的预算 |
| `videoBaseUrl` | `.../services/aigc/video-generation/video-synthesis` | 视频提交接口；任务查询地址由它派生 |
| `defaultVideoModel` | `wan3.0-video` | 视频未指定模型时使用 |
| `defaultVideoResolution` | `720P` | 视频未指定分辨率时使用 |
| `defaultVideoRatio` | `16:9` | 视频未指定比例时使用 |
| `defaultVideoDuration` | `5` | 视频未指定时长（秒）时使用 |
| `videoTimeoutMs` | `180000` | 单次提交/查询/下载请求的预算 |
| `videoPollIntervalMs` | `5000` | 轮询间隔 |
| `videoWaitMs` | `300000` | 单次工具调用默认等待时长，超时返回 task_id |
| `outputDir` | 空（用会话目录） | 会话目录不可用时的兜底目录 |

## 说明与限制

- 生成会把提示词（以及图生图/参考生视频时的输入素材）发送到千问 Token Plan 接口，按套餐用量计费；视频消耗高于图片。
- 接口返回的图片/视频链接有时效，插件会立即下载到本地，因此不依赖该链接。
- 图片只有在当前模型声明支持图片输入时才作为图片块进入对话；纯文本模型只会收到文件路径。视频只返回本地路径。
- 关闭插件后工具立即消失，已下载的文件保留在工作目录中。

## 接口参考

- [接入图像生成模型](https://platform.qianwenai.com/docs/token-plan/best-practices/image-generation)
- [接入视频生成模型](https://platform.qianwenai.com/docs/token-plan/best-practices/video-generation)
- [wan3.0-video 视频生成 API](https://platform.qianwenai.com/docs/api-reference/video-generation)
- [Qwen 图像生成与编辑 API](https://platform.qianwenai.com/docs/api-reference/image-generation/qwen-image-editing)
