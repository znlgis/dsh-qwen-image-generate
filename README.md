# 千问生图（@dsh-gis/dsh-qwen-image-generate）

给 DSH 增加一个 `qwen_image_generate` 工具：用千问 Token Plan 的生图模型按文字描述生成图片，
图片下载到当前会话的工作目录，并在对话里直接显示。

## 用法

在对话里直接说「用千问画一张……」，模型会调用该工具。也可以显式指定模型与尺寸：

- 模型：`qwen-image-3.0-pro`（默认支持文生图与图生图）、`qwen-image-2.0-pro`、`qwen-image-2.0`、
  `wan2.7-image`（默认）、`wan2.7-image-pro`
- 尺寸：`宽*高`，例如 `1024*1024`（默认）、`1536*1024`、`720*1280`
- 一次最多 6 张（`count`）；传入 `image_urls` 可让支持图生图的模型改图

生成的文件保存在会话目录下的 `qwen-images/`，命名形如 `qwen_20261125_140901_1.png`。
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
| `apiKeyEnv` | `QWEN_TOKEN_PLAN_API_KEY` | 凭据名，不是密钥值 |
| `baseUrl` | `https://token-plan.maas.qianwenaiapi.com/api/v1/services/aigc/multimodal-generation/generation` | 生成接口 |
| `defaultModel` | `wan2.7-image` | 未指定模型时使用 |
| `defaultSize` | `1024*1024` | 未指定尺寸时使用 |
| `outputDir` | 空（用会话目录） | 会话目录不可用时的兜底目录 |
| `timeoutMs` | `180000` | 单次生成与下载的预算 |

## 说明与限制

- 生成会把提示词（以及图生图时的输入图 URL）发送到千问 Token Plan 接口，按套餐的图片数量计费。
- 生图接口返回的图片链接有时效（约 24 小时），插件会立即下载到本地，因此不依赖该链接。
- 图片只有在当前模型声明支持图片输入时才作为图片块进入对话；纯文本模型只会收到文件路径。
- 关闭插件后工具立即消失，已下载的图片文件保留在工作目录中。

## 接口参考

- [接入图像生成模型](https://platform.qianwenai.com/docs/token-plan/best-practices/image-generation)
- [Qwen 图像生成与编辑 API](https://platform.qianwenai.com/docs/api-reference/image-generation/qwen-image-editing)
