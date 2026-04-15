# 小千机器人控制中心（ASR + SV + LLM + TTS + 3D 仿真）

本项目是一个本地实时语音机器人系统，包含两条可切换链路：

1. 语音控制链路：语音指令驱动 3D 小车动作（前进/后退/转向/连续运动）。
2. 语音问答链路：唤醒词 + 声纹门禁 + 大模型问答 + TTS 播报。


---

## 技术栈

- 语言与运行时：Python 3.10
- GUI：PyQt5
- 3D 可视化：pyqtgraph.opengl + OpenGL
- 音频采集：PyAudio
- 语音活动检测：webrtcvad
- ASR：FunASR（SenseVoiceSmall）
- 声纹识别：ModelScope Pipeline（CAM++）
- LLM：Transformers（Qwen2.5-1.5B-Instruct）
- TTS：edge-tts + pygame
- 语言识别：langid
- 拼音处理：pypinyin
- 日志：自研 runtime_logger（JSONL）

---

## 1. 项目目标与当前能力

### 1.1 目标

- 低时延语音控制机器人运动。
- 支持唤醒词与声纹门禁，降低误触发与非授权操作。
- 支持中/英/日基本问答播报。
- 具备结构化日志，便于复现和排障。

### 1.2 当前能力（按代码实装）

- 复合控制指令：例如“左转前进三步”。
- 连续运动：只说“前进/后退”会持续执行，直到“停止”。
- 控制模式支持误识别容错词（例如“前径”“钱进”等）。
- 控制模式声纹门禁（并带短时通过缓存，减少重复拒绝）。
- 问答模式由 15.1 子进程独立承载，GUI 直通显示子进程日志。
- 问答支持快速回复（时间/日期/天气等）优先，降低 LLM 延迟。
- TTS 回采抑制，降低“自己听到自己”的概率。

---

## 2. 项目结构（逐文件说明）

```text
ASR-LLM-TTS-master/
├─ RobotGui.py
├─ voice_engine.py
├─ 15.1_SenceVoice_kws_CAM++.py
├─ command_parser.py
├─ runtime_logger.py
├─ model.py
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ assets/
│  └─ README.txt
├─ output/                    # 运行时 ASR 音频切段
├─ Test_QWen2_VL/             # 运行时 TTS 音频输出
├─ logs/                      # JSONL 运行日志
└─ SpeakerVerification_DIR/
   └─ enroll_wav/
      └─ enroll_0.wav         # 声纹注册文件
```

### 2.1 核心源文件职责

- `RobotGui.py`
  - Qt 主窗口、3D 小车仿真、模式切换、日志显示。
  - 控制模式使用 `VoiceEngine(QThread)`。
  - 问答模式使用 `QProcess` 拉起 `15.1_SenceVoice_kws_CAM++.py`。
- `voice_engine.py`
  - GUI 内语音引擎：录音/VAD/ASR/SV/控制指令发射。
  - 内置 `chat` 分支能力，但 GUI 问答按钮当前默认使用 15.1 子进程。
- `15.1_SenceVoice_kws_CAM++.py`
  - 独立问答链路脚本：唤醒词 + 声纹 + LLM + TTS 全流程。
- `command_parser.py`
  - 控制文本解析为动作序列。
- `runtime_logger.py`
  - 输出结构化 JSONL 日志并同步打印简版终端日志。
- `model.py`
  - 占位文件（当前 `pass`）。

---

## 3. 双链路架构与模式关系

### 3.1 语音控制模式（GUI 内线程）

链路：

`Mic -> VAD/能量门限 -> 切段 wav -> ASR -> (可选唤醒) -> 声纹 -> 指令解析 -> 小车执行`

关键特点：

- 控制与问答模式互斥（避免抢麦）。
- 已注册声纹后可直接下达控制指令。
- 控制声纹含“短口令阈值”和“短时通过缓存”。
- 当前策略：停止指令允许走紧急通道（不验声纹）以保证可及时刹停。

### 3.2 语音问答模式（GUI 外子进程）

链路：

`Mic -> VAD -> ASR -> 唤醒词 -> 声纹 -> fast_reply/LLM -> TTS`

关键特点：

- 由 `15.1` 脚本独立执行。
- GUI 仅做日志直通显示（去 ANSI 控制符，不改业务文本）。
- 问答链路默认持续要求唤醒词“小千”。

### 3.3 模式互斥策略

- 打开控制模式会先停止问答子进程。
- 打开问答模式会先暂停控制引擎。
- 目的是保证同一时刻只有一个语音采集主循环在运行。

---

## 4. 详细运行流程（从启动到执行）

## 4.1 GUI 控制模式

1. 启动 `RobotGui.py`。
2. 点击“语音控制模式”。
3. `VoiceEngine` 启动并加载模型：
   - SenseVoice
   - CAM++
   - Qwen2.5（尽管控制链路主要用不到 LLM，也会被加载）
4. 语音切段通过后生成 `asr_done`。
5. 控制文本预解析：
   - 若不是控制语句，`control_non_command_drop`。
6. 声纹门禁：
   - 缓存命中：`sv_cache_hit`。
   - 缓存未命中：`sv_result`。
   - 失败：`filtered_by_sv`，不执行。
7. 通过后发 `control_text_emit`。
8. `RobotGui.on_control_text` 执行动作并输出 `control_done`。

## 4.2 GUI 问答模式（15.1 子进程）

1. 点击“语音问答模式”。
2. GUI 启动子进程：`python 15.1_SenceVoice_kws_CAM++.py`（由 `QProcess` 拉起）。
3. 子进程输出日志直接显示在 GUI 日志框。
4. 问答门禁顺序：
   - 唤醒词
   - 声纹
   - fast_reply/LLM
5. TTS 播放后进入短暂抑制窗口，减少回采。

## 4.3 声纹注册流程

- 未检测到 `enroll_0.wav` 时，命中唤醒词后进入注册态。
- 多个语音段累计到 >=3 秒后落盘注册文件。
- 注册超时或累计不足时，提示重试。

---

## 5. 关键参数总表（当前代码值）

### 5.1 `voice_engine.py` 参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `AUDIO_RATE` | 16000 | 采样率 |
| `CHUNK` | 1024 | 采集块大小 |
| `VAD_MODE` | 3 | VAD 灵敏度 |
| `VAD_WINDOW_SECONDS` | 0.4 | VAD 窗长 |
| `NO_SPEECH_THRESHOLD` | 0.58 | 句尾静音判定 |
| `MIN_UTTERANCE_SECONDS` | 0.7 | 通用最短语句时长 |
| `MIN_UTTERANCE_SECONDS_CONTROL` | 0.45 | 控制模式最短语句时长 |
| `MAX_UTTERANCE_SECONDS` | 8.0 | 单句最大时长 |
| `MIN_ENERGY` | 180.0 | 通用能量门限 |
| `MIN_ENERGY_CONTROL` | 120.0 | 控制模式能量门限 |
| `enroll_min_seconds` | 3.0 | 声纹最小累计时长 |
| `enroll_timeout_seconds` | 8.0 | 声纹注册超时 |
| `thred_sv_control` | 0.42 | 控制常规声纹阈值 |
| `thred_sv_control_short` | 0.38 | 控制短口令阈值 |
| `thred_sv_chat` | 0.50 | 问答阈值（voice_engine内chat分支） |
| `control_sv_cache_seconds` | 10.0 | 控制声纹通过缓存时长 |
| `tts_guard_seconds` | 0.45 | TTS播放后抑制窗口 |

### 5.2 `15.1_SenceVoice_kws_CAM++.py` 参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `AUDIO_RATE` | 16000 | 采样率 |
| `CHUNK` | 1024 | 采集块大小 |
| `VAD_MODE` | 3 | VAD 灵敏度 |
| `VAD_WINDOW_SECONDS` | 0.4 | VAD 窗长 |
| `NO_SPEECH_THRESHOLD` | 0.65 | 句尾静音判定 |
| `MIN_UTTERANCE_SECONDS` | 0.8 | 最短语句时长 |
| `ENROLL_MIN_SECONDS(累计)` | 3.0 | 声纹累计最小时长（分段累计） |
| `set_KWS` | `xiao qian` | 主唤醒词 |
| `kws_aliases` | `xiao qian/xiaoqian/小千/...` | 唤醒容错词 |
| `thred_sv` | 0.45 | 问答声纹阈值（较严格） |
| `require_kws_after_enroll` | True | 注册后仍需唤醒词 |

### 5.3 模型硬编码路径

当前代码中模型路径是硬编码（需本机存在）：

- `D:\AI_Models\SenseVoiceSmall`
- `D:\AI_Models\Qwen2.5-1.5B-Instruct`

若路径不一致，请先修改代码对应常量。

---

## 6. 指令解析规则（`command_parser.py`）

### 6.1 支持动作

- `forward`（前进）
- `backward`（后退）
- `left`（左转）
- `right`（右转）
- `stop`（停止）

### 6.2 数量解析

- 支持阿拉伯数字与中文数字（如“3步”“三步”）。
- 未给数量时：
  - 前进/后退可进入连续模式。
  - 转向默认一次。

### 6.3 容错词与模糊匹配

当前已加入常见 ASR 误识别容错，例如：

- 前进：`钱进`、`前景`、`前镜`、`前径` 等。
- 后退：`后腿`、`后推` 等。
- 转向：`左拐`、`右拐`、`又转` 等。
- 兜底正则模糊匹配：可识别“左么转”这类夹字变体。

---

## 7. 日志体系（终端 + JSONL + GUI）

### 7.1 `runtime_logger.py` 输出格式

每条日志写入 JSONL，基础字段：

- `ts`：毫秒级时间戳
- `event`：事件名
- 其他可选字段：`text`、`latency_ms`、`message`、`result` 等

### 7.2 控制模式常见事件

- `asr_done`
- `control_non_command_drop`
- `sv_cache_hit`
- `sv_result`
- `filtered_by_sv`
- `control_text_emit`
- `control_asr_text`
- `control_done`
- `control_stop`
- `control_stop_bypass_sv`

### 7.3 问答模式常见事件（15.1）

- `asr_done`
- `ignore_no_enroll_without_kws`
- `trigger_enroll_by_kws`
- `enroll_collecting`
- `sv_result`
- `fast_reply_hit`
- `llm_infer_ms`
- `llm_done`
- `system_intro`

### 7.4 GUI 日志显示策略

- 控制模式：显示 `VoiceEngine` 的结构化事件。
- 问答模式：直通子进程输出（只去 ANSI 字符）。
- 对 `status_update` 做短时间去重，减少刷屏。

### 7.5 GUI 与 15.1 输出对照（报告版）

| 运行形态 | 主要输出位置 | 典型可见事件/文本 |
|---|---|---|
| GUI 控制模式（`RobotGui.py + voice_engine.py`） | GUI 日志框 + 终端 + `logs/*.jsonl` | `status_update`、`asr_done`、`control_asr_text`、`sv_result`、`control_text_emit`、`control_done`、`control_stop` |
| GUI 问答模式（`RobotGui.py` 拉起 `15.1`） | GUI 日志框（子进程直通） + 子进程终端 + `logs/*.jsonl` | `system_intro`、`asr_done`、`ignore_*_without_kws`、`trigger_enroll_by_kws`、`enroll_collecting`、`sv_result`、`fast_reply_hit`、`llm_infer_ms`、`llm_done` |
| 单独运行 15.1（无 GUI） | 终端 + `logs/*.jsonl` | 与上行一致，外加 `音频保存至...`、`播放完成！` 等运行态文本 |

---

## 8. 运行说明

### 8.1 安装依赖

```bash
cd ASR-LLM-TTS-master
pip install -r requirements.txt
```

说明：

- `requirements.txt` 很大，包含许多非核心包。
- 仅运行本项目核心流程时，实际主要依赖是：
  - PyQt5, pyqtgraph, PyOpenGL
  - pyaudio, webrtcvad, numpy
  - funasr, modelscope, transformers, torch
  - edge-tts, pygame
  - langid, pypinyin

### 8.2 启动 GUI

```bash
python RobotGui.py
```

### 8.3 启动独立问答脚本

```bash
python 15.1_SenceVoice_kws_CAM++.py
```

---

## 9. 使用指南

## 9.1 控制模式

1. 点击“语音控制模式”。
2. 若未注册声纹，按提示先注册。
3. 说控制指令，例如：
   - `前进`
   - `前进三步`
   - `左转前进两步`
   - `停止`

### 9.2 问答模式

1. 点击“语音问答模式”。
2. 先说唤醒词 `小千`。
3. 再说问题（中/英/日均可尝试）。

### 9.3 关于声纹文件

请注意当前 `RobotGui.py` 逻辑：

- 在 `MainWindow.__init__` 会调用 `clear_enroll_files()`。
- 在窗口关闭 `closeEvent` 也会再次清理。

这意味着 GUI 每次重启默认会清空旧注册文件。若希望跨重启保留声纹，需要修改该行为。

---

## 10. 安全策略说明（当前实现）

- 除“停止”外，控制指令默认需要声纹通过后才执行。
- “停止”当前走紧急通道，不做声纹验证，以确保可及时刹停。

如果你希望“停止也必须验声纹”，可在 `voice_engine.py -> handle_control_logic` 中移除 `is_emergency_stop_text` 直通分支。

---

## 11. 常见问题排查

### 11.1 现象：说了指令没反应

检查顺序：

1. 日志是否有 `asr_done`。
2. 如果是 `control_non_command_drop`：ASR 文本未命中指令词，可扩充 parser 同义词。
3. 如果是 `sv_result sv=no` + `filtered_by_sv`：声纹未通过。
4. 如果问答模式中无反应：确认是否先说了“小千”。

### 11.2 现象：问答延迟高

- 复杂问题会进入 LLM，CPU 下通常是主要耗时点。
- 优先使用 fast_reply 可降低时延。
- 可考虑更小模型或 GPU。

### 11.3 现象：一直被误拒绝（sv=no）

- 降低对应链路阈值：
  - 控制链路：`thred_sv_control` / `thred_sv_control_short`
  - 问答链路：`thred_sv`
- 提升注册音频质量（安静环境、连续稳定说话）。

### 11.4 现象：问答模式日志与终端不一致

- 当前 GUI 采用 `QProcess` 直接启动问答脚本并做日志直通。
- 仅保留 ANSI 清理与状态类短时去重，不改业务文本语义。

---

## 12. 当前已知限制

- 模型路径硬编码，不方便跨机器迁移。
- `requirements.txt` 依赖体量大，环境搭建成本高。
- `15.1` 中存在部分未使用导入（例如 Qwen2VL 相关）。
- 代码内存在“GUI链路 + 子进程链路”双实现，维护复杂度较高。
- 声纹阈值与环境噪声、说话状态高度相关，需按现场调参。



## 13. 许可证

见 `LICENSE`。

---

## 14. 附录A：关键函数索引（按文件）

### 14.1 `RobotGui.py`

- `CarState.move/reset`
- `RobotVisualizer.update_pose/_on_anim_tick/_refresh_dynamic_parts`
- `MainWindow.init_ui/init_voice_engine`
- `MainWindow.toggle_control_mode/toggle_chat_mode`
- `MainWindow.start_qna_process/stop_qna_process/on_qna_output`
- `MainWindow.on_control_text`（动作执行主入口）
- `MainWindow.on_engine_runtime_event`（控制事件日志）

### 14.2 `voice_engine.py`

- `VoiceEngine.load_models/run`
- `VoiceEngine.process_audio_segment`
- `VoiceEngine.handle_control_logic`
- `VoiceEngine.handle_chat_logic`
- `VoiceEngine.fast_reply`
- `VoiceEngine.detect_speaker_lang`
- `VoiceEngine.is_kws_triggered/build_kws_candidate`
- `VoiceEngine.is_echo_text`

### 14.3 `15.1_SenceVoice_kws_CAM++.py`

- `audio_recorder/save_audio_video`
- `Inference`（问答核心主流程）
- `fast_reply`
- `detect_speaker_lang`
- `is_kws_triggered/build_kws_candidate`
- `system_introduction`
- `inference_worker`

### 14.4 `command_parser.py`

- `CommandParser.parse_sequence`（当前控制解析主入口）
- `CommandParser._scan_actions_in_order`
- `CommandParser._scan_actions_fuzzy`（ASR 形变兜底）
- `CommandParser._parse_number`

### 14.5 `runtime_logger.py`

- `RuntimeLogger.log`

---

## 15. 附录B：控制关键词与事件字典

### 15.1 控制关键词（当前实现）

- 前进：`向前走`、`往前走`、`向前进`、`往前进`、`前进`、`往前`、`向前`、`前移`、`钱进`、`前劲`、`前景`、`前镜`、`前近`、`前径`
- 后退：`向后退`、`往后退`、`向后走`、`往后走`、`后退`、`往后`、`向后`、`后移`、`后撤`、`退`、`后腿`、`后推`
- 左转：`左转`、`向左`、`往左`、`左拐`、`左边转`、`左传`
- 右转：`右转`、`向右`、`往右`、`右拐`、`右边转`、`又转`
- 停止：`停止`、`停下`、`停住`、`站住`、`别动`、`刹车`、`别跑了`、`不要跑了`、`stop`、`halt`、`停`

### 15.2 控制模式事件字典（GUI可见）

- `status_update`：状态变更文本
- `asr_done`：ASR 完成
- `control_non_command_drop`：识别文本不属于控制指令
- `sv_cache_hit`：命中 SV 短时缓存
- `sv_result`：本次 SV 判定结果
- `filtered_by_sv`：SV 未通过，已阻断执行
- `control_text_emit`：已通过门禁，准备执行
- `control_asr_text`：解析输入文本
- `control_done`：动作执行完成
- `control_stop`：停止动作落地
- `control_stop_bypass_sv`：停止走紧急通道（不验声纹）

### 15.3 问答模式事件字典（15.1）

- `system_intro`：系统播报
- `asr_done`：ASR 完成
- `ignore_no_enroll_without_kws`：未注册且无唤醒词，忽略
- `ignore_without_kws`：已注册但无唤醒词，忽略
- `trigger_enroll_by_kws`：命中唤醒词，开始注册
- `enroll_collecting`：注册累计时长
- `enroll_failed_too_short`：注册失败（时长不足）
- `sv_result`：声纹比对结果
- `filtered_by_sv`：声纹失败阻断
- `fast_reply_hit`：快速回复命中
- `llm_infer_ms`：LLM 推理耗时
- `llm_done`：LLM 输出完成
- `tts_done`：播报完成
