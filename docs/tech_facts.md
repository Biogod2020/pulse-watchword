# Pulse 技术实现事实（供总结使用，2026-11-01）

代码约 1,500 行 Python，仓库 `pulse/`，本地 git，5 项离线测试通过。

## 代码结构
- `pulse/providers.py`：模型提供方接口。
  - `Question(id, ask, options, kind=choice|noul|span, requires, scope, criteria, threshold)`；`Decision(id, choice, probs)`；`DecisionBatch(decisions, latency_ms, provider, calibrated)`。
  - `DjevDecider`：调用 djev `POST /v1/systemone`，问题类型 noul（是/否概率）、choice（各选项概率，可带每个选项的定义）、span（原文摘录，不能生成描述）；每次最多 10 题（超过会改变答案格式，按 10 题分块并发）；可带 1 张图（data URL）。
  - `OpenAICompatDecider`：任意兼容 OpenAI 的聊天接口做判断，只返回标签（calibrated=False），用于对照。
  - `ScriptedDecider`：离线测试用。
  - `Tool(name, description, parameters JSON Schema, handler)`；`OpenAICompatAgent.run(system, user, tools, max_steps=4, stop_after_tools=False)`：标准工具调用循环；stop_after_tools 表示工具执行完就返回，省一次往返（记录类功能用）。
- `pulse/heartbeat.py`：运行循环。
  - `Event(kind=speech|vision|timer, t, seg_id, rev, speaker, text, data)`。
  - `Capability(name, triggers, run, threshold=0.5, on=(事件类型), kind, dynamic)`：triggers 中任一题触发即唤醒；dynamic 是"从工作记忆生成额外问题"的函数。
  - `Rule(name, check)`：确定性规则，每个 tick 最后运行，产生 `Alert(id, severity, text, evidence)`。
  - `Heartbeat.on_event(ev)` 步骤：① perceive 钩子（如给画面写描述）② 写入工作记忆（语音修订会作废旧事实）③ 汇总所有可用功能的问题（含动态生成的），一次 decide 调用（画面事件附带图片）④ 按每题阈值判断哪些功能被唤醒 ⑤ 被唤醒的功能和到期的重试并发执行 ⑥ 规则兜底 ⑦ remember 钩子写长期记忆（失败不影响本次判断）⑧ 生成 tick 轨迹（耗时分项、每题概率、唤醒结果、告警变化、记忆版本）。
  - 失败重试：功能失败时记一条 `pending.*` 事实并在之后的 tick 重试，最多 3 次；对应语音已被修订则丢弃重试。判断器异常时规则照常运行。
- `pulse/memory.py`：工作记忆。
  - `Segment(seg_id, rev, speaker, text, t)`；`Evidence(kind=speech|vision|rule, ref, rev, quote)`；`Fact(key, value, evidence, source=observed|proposed|derived, t, version)`。
  - `add_segment`：新版本覆盖旧版本，并删除证据指向旧版本的事实（返回被作废的 key）。
  - `put`：提交事实前检查证据是否仍是最新版本，过期则拒绝（防止慢请求的旧结果覆盖新状态）。
  - 目前全部在内存。
- `pulse/longterm.py`：长期记忆。
  - `Episode(id, kind=speech|frame, t, text, image)`；`RemoteEmbedder` 调用 `tools/embed_server.py`（EmbeddingGemma 2，FastAPI，`POST /embed`，float32，768 维）。
  - 检索：语音和画面分开排名、各取前 k 条（文字对文字的相似度天然高于文字对图像，混排会把画面挤掉）。
  - 目前全部在内存。
- `pulse/packs/daily.py`：日常功能包。
  - 判断题：task_new、task_change、assistant（阈值 0.8）、put_down（图像）；动态问题：每个待办任务一个"画面里是否有人在做这件事"（最多 6 个）。
  - 功能：tasks（模型调用 add_task/update_task）、note_item（djev 选择题判断物品种类后记录位置）、task_seen（看到在做就销账）、answer（模型调用 list_open_tasks / item_locations / search_memory）。
  - perceive：djev 生成接口 `/v1/raw/chat/completions` 带图写一句描述；remember：语音和画面写入长期记忆。
- `pulse/packs/surgical_count.py`：手术室纱条清点示例包（账本、语音与画面交叉核对、关腹前兜底规则）。
- `eval_triggers.py` + `scenarios/trigger_labels.json`：判断题设计评测（30 句带陷阱的文字标注集）。
- `run_daily.py`、`run_scenario.py`：场景回放并保存轨迹 JSON。

## 实测数据
- djev 文字判断：每次 50–110 ms（单张 H100，5–6 题一次读取）；带图判断约 0.2 秒；描述 0.3–1 秒（首次调用预热约 15 秒）。
- 5 种问题设计对比（30 句）：最初合并式 8 处会导致错误动作的误判；6 个原子问题一次读 12 处（宽泛的"是否关于纱布"问题漏 10/16）；分层 4 处、44 次读取、p50 109 ms；**5 个具体问题一次读 3 处（助手阈值设 0.8 后 1 处）、p50 52 ms** ← 采用。
- 4 帧合成画面：是否放下物品、哪件物品、是否浇花全部判对。
- 日常合成场景（12 个事件）：任务新建/改期/不重复添加、三件物品位置、看到浇花自动销账、三个提问全部正确。
- EmbeddingGemma 2：nodecw1 CPU 上约 2–2.6 秒/张图；跨模态检索："glasses"→眼镜那帧第一，"my keys"→钥匙那帧第一；整句问题时文字条目会排在图片前（已用分模态排名解决）。
- 模型：DeepSeek v4.1 替身每次 2–8 秒，免费额度会 429；gpt-6-luna（CLIProxyAPI）工具调用 2.3 秒、看图 8.9 秒，不支持 /v1/decisions。

## 待实现的技术细节
- 记忆持久化：SQLite 表 segments、frames、facts、evidence、episodes、tasks；向量先 numpy 后 sqlite-vec/FAISS，EmbeddingGemma 2 的 256 维压缩版；图片按 sha256 存文件；L1 滚动 14 天、上限约 1 GB；L2 证据帧随事实失效替换；整理命令把当天情景压成文字摘要。
- 去重：浏览器侧灰度缩略图像素差；后端向量相似度 >0.95 视为重复。
- 调度：画面高频时只处理最新关键帧；语音/提问优先；长期记忆写入改后台任务。
- 模型配置：PULSE_MODEL_BASE_URL / PULSE_MODEL / PULSE_MODEL_KEY_FILE，dev=gpt-6-luna（CLIProxyAPI 127.0.0.1:8317），prod=muse-spark-1.3-contributor（Meta Model API）；结构化输出返回证据帧上的物品框坐标。
- 界面：FastAPI + 静态页面 + WebSocket；消息类型 event（上行）、tick、memory、answer（下行）。
- MCP：官方 Python MCP 客户端接 Google 日历（可选）。
- 部署：djev 在 gpucw1 GPU0（vLLM 0.32、AWQ INT4、FP8 KV、每次最多 1 张图），嵌入服务在 nodecw1，应用在 headcw，SSH 隧道连接。
