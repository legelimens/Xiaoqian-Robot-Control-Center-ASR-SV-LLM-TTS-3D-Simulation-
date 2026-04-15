import os
import sys
import time
import wave
import asyncio
import re
from collections import deque
from difflib import SequenceMatcher

import pyaudio
import webrtcvad
import numpy as np
import pygame
import edge_tts
import langid
import torch
from pypinyin import pinyin, Style
from transformers import AutoModelForCausalLM, AutoTokenizer
from funasr import AutoModel
from modelscope.pipelines import pipeline
from PyQt5.QtCore import QThread, pyqtSignal, QObject

from runtime_logger import RuntimeLogger
from command_parser import CommandParser

# --- 配置huggingFace国内镜像 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class VoiceSignals(QObject):
    """信号定义"""
    asr_result = pyqtSignal(str)      # 语音识别结果
    llm_result = pyqtSignal(str)      # LLM 生成结果
    control_text = pyqtSignal(str)    # 控制模式通过声纹校验的文本
    runtime_event = pyqtSignal(str, str)  # 运行时事件日志（event, text）
    status_update = pyqtSignal(str)   # 状态更新提示
    error_occurred = pyqtSignal(str)  # 错误信息


class ChatMemory:
    def __init__(self, max_chars=1024):
        self.history = []
        self.max_chars = max_chars

    def add_turn(self, user_input, model_response):
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": model_response})
        self._truncate()

    def _truncate(self):
        def total_chars():
            return sum(len(x.get("content", "")) for x in self.history)

        while self.history and total_chars() > self.max_chars:
            self.history.pop(0)

    def get_messages(self):
        return list(self.history)


class VoiceEngine(QThread):
    def __init__(self):
        super().__init__()
        self.signals = VoiceSignals()
        self.logger = RuntimeLogger()

        # 参数设置
        self.AUDIO_RATE = 16000
        self.CHUNK = 1024
        self.VAD_MODE = 3
        self.VAD_WINDOW_SECONDS = 0.4
        self.OUTPUT_DIR = os.path.join(BASE_DIR, "output")
        self.NO_SPEECH_THRESHOLD = 0.58
        self.MIN_UTTERANCE_SECONDS = 0.7
        self.MIN_UTTERANCE_SECONDS_CONTROL = 0.45
        self.MAX_UTTERANCE_SECONDS = 8.0
        self.MIN_ENERGY = 180.0
        self.MIN_ENERGY_CONTROL = 120.0
        self.MAX_FILES = 10
        self.folder_path = os.path.join(BASE_DIR, "Test_QWen2_VL")

        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.folder_path, exist_ok=True)

        self.recording_active = False
        self.mode = "none"
        self.audio_file_count = 0
        self.enroll_min_seconds = 3.0
        self.enroll_timeout_seconds = 8.0
        self.enroll_started_at = 0.0
        self.enroll_accum_frames = []
        self.enroll_accum_seconds = 0.0
        self.require_sv_for_control = True
        self.require_sv_for_chat = True

        # 唤醒词与声纹配置
        self.set_KWS = "xiao qian"
        self.kws_aliases = [
            "xiao qian",
            "xiaoqian",
            "小千",
            "小倩",
            "小钱",
            "晓倩",
        ]
        self.kws_recent = deque(maxlen=6)
        self.last_kws_hint_ts = 0.0
        self.require_kws_after_enroll = True
        self.set_SV_enroll = os.path.join(BASE_DIR, "SpeakerVerification_DIR", "enroll_wav")
        # 分场景阈值：控制模块略放宽，避免已注册用户被频繁误拒
        self.thred_sv_control = 0.42
        self.thred_sv_control_short = 0.38
        self.thred_sv_chat = 0.50
        self.control_sv_cache_seconds = 10.0
        self.control_sv_pass_until = 0.0
        os.makedirs(self.set_SV_enroll, exist_ok=True)

        # 自说自听抑制
        self.tts_guard_seconds = 0.45
        self.tts_busy_until = 0.0
        self.last_tts_text = ""
        self.last_tts_ts = 0.0

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(self.VAD_MODE)

        self.memory = ChatMemory(max_chars=1024)
        self.models_loaded = False
        self.mode_before_enroll = "none"
        self.control_parser = CommandParser()

    def load_models(self):
        """加载所有 AI 模型"""
        try:
            self.signals.status_update.emit("正在加载 AI 模型，请稍候...")
            self.logger.log("model_loading_start")

            # 1. SenseVoice
            model_dir = r"D:\AI_Models\SenseVoiceSmall"
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"找不到模型目录: {model_dir}")

            if model_dir not in sys.path:
                sys.path.append(model_dir)

            try:
                self.model_senceVoice = AutoModel(
                    model=model_dir,
                    trust_remote_code=True,
                    disable_update=True
                )
            except Exception as e:
                msg = str(e)
                if "No module named 'model'" in msg or "Loading remote code failed" in msg:
                    self.signals.status_update.emit("本地 SenseVoice 缺少代码文件，尝试在线加载 iic/SenseVoiceSmall...")
                    self.model_senceVoice = AutoModel(
                        model="iic/SenseVoiceSmall",
                        trust_remote_code=True,
                        disable_update=True
                    )
                else:
                    raise

            # 2. CAM++
            self.sv_pipeline = pipeline(
                task='speaker-verification',
                model='damo/speech_campplus_sv_zh-cn_16k-common',
                model_revision='v1.0.0'
            )

            # 3. Qwen2.5
            model_name = r"D:\AI_Models\Qwen2.5-1.5B-Instruct"
            if not os.path.exists(model_name):
                raise FileNotFoundError(f"找不到模型目录: {model_name}")

            self.model_llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            self.model_llm.to("cpu")
            self.model_llm.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            try:
                self.model_llm.generation_config.temperature = None
                self.model_llm.generation_config.top_p = None
                self.model_llm.generation_config.top_k = None
            except Exception:
                pass

            self.models_loaded = True
            self.signals.status_update.emit("模型加载完成，准备就绪！")
            self.logger.log("model_loading_done")
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            self.logger.log("model_loading_failed", message=error_msg)
            self.signals.error_occurred.emit(error_msg)
            self.signals.status_update.emit("模型加载失败，请检查路径或环境")
            self.recording_active = False

    def run(self):
        """线程主循环：处理录音与推理"""
        try:
            if not self.models_loaded:
                self.load_models()

            if not self.models_loaded:
                return

            self.recording_active = True
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.AUDIO_RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
            )

            audio_buffer = []
            buffer_bytes = 0
            segments_to_save = []
            last_active_time = time.time()
            window_target_bytes = int(self.AUDIO_RATE * 2 * self.VAD_WINDOW_SECONDS)

            enroll_path = os.path.join(os.path.abspath(self.set_SV_enroll), "enroll_0.wav")
            if os.path.exists(enroll_path):
                self.signals.status_update.emit("您已开启声纹识别和关键词唤醒，请说“小千”后再提问。")
            else:
                self.signals.status_update.emit("您已开启声纹识别和关键词唤醒，目前无声纹注册文件！请先注册声纹，需大于三秒哦~")
            self.logger.log("audio_loop_start")

            while self.recording_active:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)

                    now = time.time()
                    if (
                        self.mode == "enroll"
                        and self.enroll_started_at > 0
                        and (now - self.enroll_started_at) > self.enroll_timeout_seconds
                        and self.enroll_accum_seconds < self.enroll_min_seconds
                        and (now - last_active_time) > self.NO_SPEECH_THRESHOLD
                    ):
                        fail_msg = "声纹注册失败，需大于三秒，请再次说“小千”重新注册。"
                        self.signals.status_update.emit(fail_msg)
                        self.logger.log(
                            "enroll_failed_too_short",
                            duration_s=round(self.enroll_accum_seconds, 2),
                            message=fail_msg
                        )
                        restore_mode = self.mode_before_enroll if self.mode_before_enroll in ("chat", "control") else "control"
                        self.mode = restore_mode
                        self.enroll_started_at = 0.0
                        self.enroll_accum_seconds = 0.0
                        self.enroll_accum_frames = []
                        self.logger.log("enroll_mode_restore", message=f"mode={restore_mode}")

                    if now < self.tts_busy_until:
                        # TTS 播放窗口内直接忽略采集，避免自说自听
                        audio_buffer.clear()
                        buffer_bytes = 0
                        segments_to_save.clear()
                        continue

                    audio_buffer.append(data)
                    buffer_bytes += len(data)

                    if buffer_bytes >= window_target_bytes:
                        raw_audio = b''.join(audio_buffer)
                        audio_buffer = []
                        buffer_bytes = 0

                        vad_active = self.check_vad_activity(raw_audio)
                        energy = self.compute_energy(raw_audio)
                        energy_thr = self.MIN_ENERGY_CONTROL if self.mode == "control" else self.MIN_ENERGY
                        min_utt_thr = self.MIN_UTTERANCE_SECONDS_CONTROL if self.mode == "control" else self.MIN_UTTERANCE_SECONDS

                        if vad_active and energy >= energy_thr:
                            ts = time.time()
                            last_active_time = ts
                            segments_to_save.append((raw_audio, ts))

                        if segments_to_save:
                            duration_s = len(segments_to_save) * self.VAD_WINDOW_SECONDS
                            silence_t = time.time() - last_active_time
                            if silence_t > self.NO_SPEECH_THRESHOLD or duration_s >= self.MAX_UTTERANCE_SECONDS:
                                if duration_s >= min_utt_thr:
                                    self.process_audio_segment(segments_to_save)
                                else:
                                    self.logger.log("segment_dropped_short", duration_s=round(duration_s, 2), threshold=min_utt_thr)
                                segments_to_save = []
                                last_active_time = time.time()

                except Exception as e:
                    self.signals.error_occurred.emit(f"录音异常: {str(e)}")
                    self.logger.log("audio_loop_error", message=str(e))
                    break

            stream.stop_stream()
            stream.close()
            p.terminate()
            self.logger.log("audio_loop_stop")
        except Exception as e:
            self.signals.error_occurred.emit(f"线程崩溃: {str(e)}")
            self.logger.log("thread_crash", message=str(e))

    def check_vad_activity(self, audio_data):
        num = 0
        rate = 0.5
        step = int(self.AUDIO_RATE * 0.02) * 2  # 20ms * 16bit
        frame_count = max(1, len(audio_data) // step)
        flag_rate = round(rate * frame_count)

        for i in range(0, len(audio_data), step):
            chunk = audio_data[i:i + step]
            if len(chunk) == step and self.vad.is_speech(chunk, sample_rate=self.AUDIO_RATE):
                num += 1

        return num > flag_rate

    def compute_energy(self, audio_data):
        try:
            pcm = np.frombuffer(audio_data, dtype=np.int16)
            if pcm.size == 0:
                return 0.0
            return float(np.mean(np.abs(pcm)))
        except Exception:
            return 0.0

    def _write_wav(self, path, audio_frames):
        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.AUDIO_RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()

    def process_audio_segment(self, segments):
        """保存并处理音频段"""
        if self.mode == "none":
            return
        if time.time() < self.tts_busy_until:
            return

        duration_s = len(segments) * self.VAD_WINDOW_SECONDS
        segment_end_ts = segments[-1][1]

        if self.mode == "enroll":
            self.enroll_accum_frames.extend([seg[0] for seg in segments])
            self.enroll_accum_seconds += duration_s
            self.signals.status_update.emit(
                f"声纹注册中：{self.enroll_accum_seconds:.1f}s / {self.enroll_min_seconds:.1f}s"
            )
            self.logger.log("enroll_collecting", duration_s=round(self.enroll_accum_seconds, 2))

            if self.enroll_accum_seconds < self.enroll_min_seconds:
                return

            enroll_path = os.path.join(os.path.abspath(self.set_SV_enroll), "enroll_0.wav")
            self._write_wav(enroll_path, self.enroll_accum_frames)
            self.signals.status_update.emit("声纹注册完成")
            self.logger.log("enroll_done", path=enroll_path, duration_s=round(self.enroll_accum_seconds, 2))
            self.enroll_accum_frames = []
            self.enroll_accum_seconds = 0.0
            self.enroll_started_at = 0.0
            self.kws_recent.clear()
            self.mode = self.mode_before_enroll if self.mode_before_enroll in ("chat", "control") else "control"
            self.logger.log("enroll_mode_restore", message=f"mode={self.mode}")
            return

        self.audio_file_count = (self.audio_file_count % self.MAX_FILES) + 1
        file_path = os.path.join(self.OUTPUT_DIR, f"audio_{self.audio_file_count}.wav")
        self._write_wav(file_path, [seg[0] for seg in segments])

        if not self.models_loaded:
            self.load_models()
            if not self.models_loaded:
                return

        asr_t0 = time.perf_counter()
        res = self.model_senceVoice.generate(input=file_path, cache={}, language="auto", use_itn=False)
        asr_ms = (time.perf_counter() - asr_t0) * 1000
        text = res[0]['text'].split(">")[-1].strip()
        if not text:
            return
        if len(self.normalize_text(text)) < 2:
            self.logger.log("asr_too_short_drop", text=text)
            return

        if self.is_echo_text(text):
            self.logger.log("echo_filtered", text=text)
            return

        self.signals.asr_result.emit(text)

        e2e_ms = (time.time() - segment_end_ts) * 1000
        self.logger.log("asr_done", text=text, asr_ms=round(asr_ms, 2), latency_ms=round(e2e_ms, 2))

        if self.mode == "chat":
            self.handle_chat_logic(text, file_path, e2e_ms)
        else:
            self.handle_control_logic(text, file_path, e2e_ms)

    def handle_chat_logic(self, text, file_path, e2e_ms):
        """聊天模式：仅在唤醒词“小千”触发后继续执行"""
        enroll_path = os.path.join(os.path.abspath(self.set_SV_enroll), "enroll_0.wav")
        has_enroll = os.path.exists(enroll_path)
        kws_text = self.build_kws_candidate(text)
        kws_hit = self.is_kws_triggered(kws_text)

        if not kws_hit:
            if not has_enroll:
                self.logger.log("ignore_no_enroll_without_kws", text=text)
                if time.time() - self.last_kws_hint_ts > 4.0:
                    self.signals.status_update.emit("等待唤醒词“小千”以开始声纹注册")
                    self.last_kws_hint_ts = time.time()
            else:
                self.logger.log("ignore_without_kws", text=text)
            return

        if not has_enroll:
            self.signals.status_update.emit("检测到唤醒词，开始录入声纹：请连续说话≥3秒")
            self.logger.log("chat_trigger_enroll", text=text)
            self.enroll_accum_frames = []
            self.enroll_accum_seconds = 0.0
            self.enroll_started_at = time.time()
            self.mode_before_enroll = "chat"
            self.mode = "enroll"
            return

        if self.require_sv_for_chat:
            sv_score = self.sv_pipeline([enroll_path, file_path], thr=self.thred_sv_chat)
            if sv_score.get("text") != "yes":
                self.logger.log("chat_filtered_by_sv", text=text, sv_result=sv_score)
                self.signals.status_update.emit("声纹识别失败，不好意思我不能为你服务")
                return

        prompt_text = self.strip_wakeword_prefix(text)
        if not prompt_text:
            output_text = "我在，你说。"
            self.logger.log("fast_reply_hit", text="wake_only")
            self.signals.llm_result.emit(output_text)
            self.signals.status_update.emit(f"问答完成：总时延 {e2e_ms:.0f}ms")
            self.tts_and_play(output_text, user_text=text)
            return

        quick_answer = self.fast_reply(prompt_text)
        if quick_answer is not None:
            output_text = quick_answer
            llm_ms = 0.0
            self.logger.log("fast_reply_hit", text=prompt_text)
        else:
            user_lang = self.detect_speaker_lang("", prompt_text)
            lang_rule = ""
            if user_lang == "en":
                lang_rule = "请使用简洁英文回答。"
            elif user_lang == "ja":
                lang_rule = "日本語で簡潔に答えてください。"
            messages = [
                {"role": "system", "content": f"你叫小千，是一个18岁的女大学生，性格活泼开朗，说话俏皮简洁，回答问题不会超过50字。{lang_rule}"},
            ]
            messages.extend(self.memory.get_messages()[-1:])
            messages.append({"role": "user", "content": prompt_text})

            llm_t0 = time.perf_counter()
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model_llm.device)
            generated_ids = self.model_llm.generate(**model_inputs, max_new_tokens=24, do_sample=False)
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            llm_ms = (time.perf_counter() - llm_t0) * 1000

        output_text = self.sanitize_model_output(output_text)
        self.memory.add_turn(prompt_text, output_text)
        self.signals.llm_result.emit(output_text)

        total_ms = e2e_ms + llm_ms
        self.signals.status_update.emit(f"问答完成：总时延 {total_ms:.0f}ms")
        self.logger.log(
            "chat_done",
            text=prompt_text,
            answer=output_text,
            llm_ms=round(llm_ms, 2),
            latency_ms=round(total_ms, 2),
        )
        self.tts_and_play(output_text, user_text=prompt_text)

    def handle_control_logic(self, text, file_path, e2e_ms):
        """控制模式：有声纹时可直接发指令，先做声纹校验再执行"""
        enroll_path = os.path.join(os.path.abspath(self.set_SV_enroll), "enroll_0.wav")
        has_enroll = os.path.exists(enroll_path)
        cmd_text = self.strip_wakeword_prefix(text)

        # 安全急停通道：停止指令不做声纹校验，确保能及时停下
        if self.is_emergency_stop_text(cmd_text or text):
            self.logger.log("control_stop_bypass_sv", text=text)
            self.signals.runtime_event.emit("control_stop_bypass_sv", cmd_text or text)
            self.signals.control_text.emit("停止")
            return

        # 仅在未注册声纹时，才要求唤醒词进入注册流程
        if not has_enroll:
            kws_text = self.build_kws_candidate(text)
            kws_hit = self.is_kws_triggered(kws_text)
            if not kws_hit:
                self.logger.log("ignore_no_enroll_without_kws", text=text)
                self.signals.runtime_event.emit("ignore_no_enroll_without_kws", text)
                if time.time() - self.last_kws_hint_ts > 4.0:
                    self.signals.status_update.emit("等待唤醒词“小千”以开始声纹注册")
                    self.last_kws_hint_ts = time.time()
                return

            self.signals.status_update.emit("检测到唤醒词，开始录入声纹：请连续说话≥3秒")
            self.logger.log("control_trigger_enroll", text=text)
            self.signals.runtime_event.emit("control_trigger_enroll", "start_enroll")
            self.enroll_accum_frames = []
            self.enroll_accum_seconds = 0.0
            self.enroll_started_at = time.time()
            self.mode_before_enroll = "control"
            self.mode = "enroll"
            return

        if not cmd_text:
            self.signals.status_update.emit("我在，你说指令。")
            return

        # 仅对“看起来像控制指令”的语句做声纹校验，可显著减少无效延迟
        preview_cmds = self.control_parser.parse_sequence(cmd_text)
        if not preview_cmds:
            self.logger.log("control_non_command_drop", text=cmd_text)
            self.signals.runtime_event.emit("control_non_command_drop", cmd_text)
            self.signals.status_update.emit("未识别到控制指令")
            return

        # 已有声纹：直接进行声纹门禁（无需唤醒词）
        if self.require_sv_for_control:
            now = time.time()
            if now <= self.control_sv_pass_until:
                remain = self.control_sv_pass_until - now
                self.logger.log("sv_cache_hit", message=f"remain={remain:.1f}s", text=text)
                self.signals.runtime_event.emit("sv_cache_hit", f"remain={remain:.1f}s")
            else:
                norm_cmd = self.normalize_text(cmd_text)
                thr = self.thred_sv_control_short if len(norm_cmd) <= 4 else self.thred_sv_control
                sv_score = self.sv_pipeline([enroll_path, file_path], thr=thr)
                sv_text = sv_score.get("text")
                score = sv_score.get("score")
                score_text = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
                self.logger.log("sv_result", message=f"sv={sv_text} thr={thr:.2f}{score_text}", result=sv_score, text=text)
                self.signals.runtime_event.emit("sv_result", f"sv={sv_text} thr={thr:.2f}{score_text}")
                self.signals.status_update.emit(f"sv_result sv={sv_text}")
                if sv_text != "yes":
                    self.logger.log("filtered_by_sv", text=text)
                    self.signals.runtime_event.emit("filtered_by_sv", text)
                    self.signals.status_update.emit("声纹识别失败，不好意思我不能为你服务")
                    return
                self.control_sv_pass_until = time.time() + self.control_sv_cache_seconds

        self.signals.status_update.emit(f"指令识别: {cmd_text} | 时延 {e2e_ms:.0f}ms")
        self.signals.runtime_event.emit("control_text_emit", cmd_text)
        self.signals.control_text.emit(cmd_text)
        self.logger.log("control_text_emit", text=cmd_text, latency_ms=round(e2e_ms, 2))

    def is_emergency_stop_text(self, text):
        t = (text or "").strip()
        if not t:
            return False
        cmds = self.control_parser.parse_sequence(t)
        if any(cmd.get("action") == "stop" for cmd in cmds):
            return True
        compact = re.sub(r"\s+", "", t.lower())
        # ASR 极端截断容错，例如“ing止”
        if compact.endswith("止") and len(compact) <= 4:
            return True
        return False

    def sanitize_model_output(self, text):
        if not text:
            return text
        text = re.sub(r'^\s*(系统|system|assistant)\s*[:：]\s*', '', text, flags=re.IGNORECASE)
        return text.strip()

    def strip_wakeword_prefix(self, text):
        t = (text or "").strip()
        t = re.sub(
            r'^\s*(你好|嗨|哈喽|hello|hi|hey|こんにちは|もしもし)?\s*[，,\s]*'
            r'(小[千倩钱]|晓倩|xiaoqian|xiao\s*qian)\s*[，,\s]*',
            '',
            t,
            flags=re.IGNORECASE
        )
        return t.strip()

    def fast_reply(self, prompt_text):
        t = (prompt_text or "").strip()
        if not t:
            return None

        # 中文时间/日期
        if ("几点" in t) or ("时间" in t):
            return f"现在是{time.strftime('%H点%M分')}"
        if ("几月几号" in t) or ("今天几号" in t) or ("今天是几号" in t) or ("日期" in t):
            return f"今天是{time.strftime('%Y年%m月%d日')}"
        if ("你在干什么" in t) or ("在干嘛" in t):
            return "我在听你说话呀，你想让我做什么？"
        if ("天气" in t):
            return "我现在无法联网查询实时天气，但可以给你一般天气建议。"

        # 英文时间/日期
        t_low = re.sub(r"[^a-z0-9\s']", " ", t.lower())
        t_low = " ".join(t_low.split())
        if ("what time" in t_low) or ("time is it" in t_low) or ("whats time" in t_low) or ("time now" in t_low):
            return f"It is {time.strftime('%H:%M')} now."
        if ("what date" in t_low) or ("today date" in t_low) or ("today s date" in t_low) or ("date today" in t_low):
            return f"Today is {time.strftime('%Y-%m-%d')}."
        if ("weather" in t_low) or ("temperature" in t_low):
            return "I cannot fetch live weather now, but I can give general weather advice."
        if ("can you hear me" in t_low):
            return "Yes, I can hear you clearly."

        # 日文时间/日期
        if ("今何時" in t) or ("何時" in t) or ("いまなんじ" in t) or ("今はいつ" in t):
            return f"今は{time.strftime('%H時%M分')}です。"
        if ("何月何日" in t) or ("今日の日付" in t) or ("きょうはなんがつなんにち" in t):
            return f"今日は{time.strftime('%Y年%m月%d日')}です。"
        return None

    def tts_and_play(self, text, user_text=""):
        """文字转语音并播放"""
        text = self.sanitize_model_output(text)
        if not text:
            return

        used_speaker = "zh-CN-XiaoyiNeural"
        try:
            lang = self.detect_speaker_lang(text, user_text)
            lang_map = {
                "ja": "ja-JP-NanamiNeural",
                "en": "en-US-AnaNeural",
                "zh": "zh-CN-XiaoyiNeural",
            }
            used_speaker = lang_map.get(lang, "zh-CN-XiaoyiNeural")
        except Exception:
            pass

        output_file = os.path.join(self.folder_path, "tts_out.mp3")
        self.last_tts_text = text
        self.last_tts_ts = time.time()

        try:
            asyncio.run(self._edge_tts_save(text, used_speaker, output_file))
            self.play_audio(output_file)
        finally:
            self.tts_busy_until = time.time() + self.tts_guard_seconds
            self.logger.log("tts_done", text=text, voice=used_speaker)

    async def _edge_tts_save(self, text, voice, path):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(path)

    def play_audio(self, path):
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            self.logger.log("audio_play_error", message=str(e))
        finally:
            try:
                pygame.mixer.quit()
            except Exception:
                pass

    def normalize_text(self, text):
        text = text or ""
        text = re.sub(r'[^\w\u4e00-\u9fa5]+', '', text.lower())
        return text

    def has_han(self, text):
        return bool(re.search(r'[\u4e00-\u9fff]', text or ""))

    def has_kana(self, text):
        return bool(re.search(r'[\u3040-\u30ff]', text or ""))

    def latin_ratio(self, text):
        if not text:
            return 0.0
        letters = re.findall(r'[A-Za-z]', text)
        return len(letters) / max(1, len(text))

    def detect_speaker_lang(self, reply_text, user_text=""):
        text = (reply_text or "").strip()
        user = (user_text or "").strip()
        if self.has_kana(user):
            return "ja"
        # 混合文本（如 xiao倩ian）优先按拉丁占比判定为英文，避免误回中文音色
        if self.latin_ratio(user) > 0.45 and not self.has_kana(user):
            return "en"
        if self.has_han(user):
            return "zh"
        if self.has_kana(text):
            return "ja"
        if self.latin_ratio(text) > 0.55 and not self.has_kana(text):
            return "en"
        if self.has_han(text):
            return "zh"
        try:
            lang, _ = langid.classify(user or text)
            return lang
        except Exception:
            return "zh"

    def is_echo_text(self, asr_text):
        # 最近 8 秒内的 TTS 文本做回采过滤
        if time.time() - self.last_tts_ts > 8:
            return False
        a = self.normalize_text(asr_text)
        b = self.normalize_text(self.last_tts_text)
        if len(a) < 4 or len(b) < 4:
            return False
        if a in b or b in a:
            return True
        return SequenceMatcher(None, a, b).ratio() >= 0.85

    def extract_pinyin(self, text):
        chinese_characters = re.findall(r'[\u4e00-\u9fa5]', text)
        chinese_text = ''.join(chinese_characters)
        pinyin_result = pinyin(chinese_text, style=Style.NORMAL)
        return ' '.join([item[0] for item in pinyin_result])

    def normalize_pinyin(self, text):
        text = (text or "").lower()
        text = re.sub(r'[^a-z0-9\u3040-\u30ff\u4e00-\u9fff\s]+', '', text)
        return ' '.join(text.split())

    def build_kws_candidate(self, text):
        py = self.extract_pinyin(text)
        en = self.normalize_pinyin(text)
        if py and en:
            return f"{py} {en}"
        return py or en

    def is_kws_triggered(self, pinyin_text):
        norm = self.normalize_pinyin(pinyin_text)
        if not norm:
            return False

        self.kws_recent.append(norm)
        joined = ' '.join(self.kws_recent)
        joined_compact = joined.replace(" ", "")

        for alias in self.kws_aliases:
            alias_norm = self.normalize_pinyin(alias)
            alias_compact = alias_norm.replace(" ", "")
            if not alias_compact:
                continue

            if alias_norm in joined or alias_compact in joined_compact:
                self.kws_recent.clear()
                return True

            tail = joined_compact[-(len(alias_compact) + 4):]
            if len(tail) >= max(4, len(alias_compact) - 3):
                if SequenceMatcher(None, tail, alias_compact).ratio() >= 0.82:
                    self.kws_recent.clear()
                    return True

        return False

    def stop(self):
        self.recording_active = False
        self.wait()
