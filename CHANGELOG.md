# 更新日志

## 1.4.0

- 生图预算拆成生成与下载两段：生成阶段超时后仍会保存已完成并计费的图片；一张都没生成时返回 `QWEN_IMAGE_TIMEOUT`。
- 只有在服务明确拒绝批量字段 `n` 时才退回逐张请求；其他 400 不再按图片数重复请求，避免重复计费。
- `negative_prompt` 超过 500 字符时截断，并在 `warnings` 里说明。
- 图片/视频下载加上限（`maxImageBytes` 默认 64 MiB、`maxVideoBytes` 默认 512 MiB），流式读取超限即中断。
- 视频下载失败会重试；下载被取消或超时时返回 `pending` 并保留 taskId，重调只下载文件，不重新渲染。
- 视频任务账本按输出目录串行更新，并发提交不再互相覆盖；账本记录完整提示词与提交参数。
- 用 `task_id` 续查时沿用账本里记录的模型，prompt/素材被忽略时写进 `warnings`。
- 视频成片同样写入输出目录的 `.manifest.jsonl`（用 `kind` 区分图片与视频）。
- `videoBaseUrl` 在提交前校验，配置错误不会再产生已计费却查不到的任务。
- 文件名前缀拒绝 Windows 保留设备名（CON、NUL 等）。
- 连续拿不到任务状态时保留 taskId 并返回 `pending`，不再删除账本条目（任务可能仍在渲染并计费）。
- 账本改为“写临时文件再改名”的原子替换；损坏的账本改名保留，不再被下一次写入静默清空。
- `output_dir` 越界检查会解析符号链接/junction；非字符串的 `output_dir` 直接报 `INVALID_ARGS`。
- 视频提交/查询的超时不再被当成网络错误重试；提交超时返回 `QWEN_VIDEO_SUBMIT_UNKNOWN`，提交预算封顶 60 秒。
- `wait_seconds` 超过宿主默认 120 秒时给出 `warnings` 提醒，并说明被宿主杀掉后如何用 task_id 续查。
- 图片全部下载失败时，错误信息带上首个源 URL；部分失败时 `warnings` 也带源 URL，便于人工抢救。
- 部分提示词失败时，`warnings` 标出是第几条提示词。
- 附件服务未返回 `mediaType` 时不再把已保存文件的类型置空；相关字段描述也更准确。
- 新增 MIT LICENSE、CHANGELOG、`.gitignore`；CI 覆盖 Node 20/22/24，并运行 `npm pack --dry-run` 校验发布内容。

## 1.3.0

- 批量提示词、文件命名前缀、manifest 账本与生图重试（见仓库 1.3.0 提交）。
