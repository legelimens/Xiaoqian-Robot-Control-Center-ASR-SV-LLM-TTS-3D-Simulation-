import cv2
import pyaudio
import wave
import threading
import numpy as np
import time
from queue import Queue
import webrtcvad
import os
import errno
if not hasattr(errno, 'EREMOTEIO'):
    setattr(errno, 'EREMOTEIO', 121)
import threading
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
from funasr import AutoModel
import pygame
import edge_tts
import asyncio
from time import sleep
import langid
from langdetect import detect
import re
from difflib import SequenceMatcher
from pypinyin import pinyin, Style
from modelscope.pipelines import pipeline
import atexit
from runtime_logger import RuntimeLogger

# --- 配置huggingFace国内镜像 ---
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 参数设置
AUDIO_RATE = 16000        # 音频采样率
AUDIO_CHANNELS = 1        # 单声道
CHUNK = 1024              # 音频块大小
VAD_MODE = 3              # VAD 模式 (0-3, 数字越大越敏感)
VAD_WINDOW_SECONDS = 0.4  # VAD 窗长，缩短可降低切句延迟
OUTPUT_DIR = os.path.join(BASE_DIR, "output")   # 输出目录
NO_SPEECH_THRESHOLD = 0.58   # 句尾静音阈值（更快收句）
MIN_UTTERANCE_SECONDS = 0.8  # 最短有效语句时长
MAX_FILES = 10            # 最大保留音频文件数
ENROLL_MIN_SECONDS = 3.0
ENROLL_TIMEOUT_SECONDS = 8.0
folder_path = os.path.join(BASE_DIR, "Test_QWen2_VL")
audio_file_count = 0
audio_file_count_tmp = 0
tts_file_seq = 0
tts_file_seq_lock = threading.Lock()
audio_playback_lock = threading.Lock()
inference_lock = threading.Lock()
inference_running = False
logger = RuntimeLogger()

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(folder_path, exist_ok=True)

# 队列用于音频和视频同步缓存
audio_queue = Queue()
video_queue = Queue()

# 全局变量
last_active_time = time.time()
recording_active = True
segments_to_save = []
saved_intervals = []
last_vad_end_time = 0  # 上次保存的 VAD 有效段结束时间
enroll_accum_frames = []
enroll_accum_seconds = 0.0
enroll_started_ts = 0.0


# --- 唤醒词、声纹变量配置 ---
set_KWS = "xiao qian"
kws_aliases = [
    "xiao qian",
    "xiaoqian",
    "小千",
    # ASR 常见误识别容错，保持“唤醒词=小千”语义不变
    "小倩",
    "小钱",
    "晓倩",
]
# set_KWS = "shuo hua xiao qian"
# set_KWS = "zhan qi lai"
flag_KWS = 0
recent_kws_pinyin = []
last_kws_hint_ts = 0.0

flag_KWS_used = 1
flag_sv_used = 1

flag_sv_enroll = 0
# 提高阈值，进一步减少误放行（更严格）
thred_sv = 0.50
tts_busy_until = 0.0
last_tts_text = ""
last_tts_ts = 0.0
require_kws_after_enroll = True

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(VAD_MODE)


def extract_chinese_and_convert_to_pinyin(input_string):
    """
    提取字符串中的汉字，并将其转换为拼音。
    
    :param input_string: 原始字符串
    :return: 转换后的拼音字符串
    """
    # 使用正则表达式提取所有汉字
    chinese_characters = re.findall(r'[\u4e00-\u9fa5]', input_string)
    # 将汉字列表合并为字符串
    chinese_text = ''.join(chinese_characters)
    
    # 转换为拼音
    pinyin_result = pinyin(chinese_text, style=Style.NORMAL)
    # 将拼音列表拼接为字符串
    pinyin_text = ' '.join([item[0] for item in pinyin_result])
    
    return pinyin_text


# 音频录制线程
def audio_recorder():
    global audio_queue, recording_active, last_active_time, segments_to_save, last_vad_end_time, tts_busy_until
    global flag_sv_enroll, enroll_accum_seconds, enroll_accum_frames, enroll_started_ts, recent_kws_pinyin
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=AUDIO_CHANNELS,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    audio_buffer = []
    print("音频录制已开始")
    
    while recording_active:
        now = time.time()
        if (
            flag_sv_enroll
            and enroll_started_ts > 0
            and (now - enroll_started_ts) > ENROLL_TIMEOUT_SECONDS
            and enroll_accum_seconds < ENROLL_MIN_SECONDS
            and (now - last_active_time) > NO_SPEECH_THRESHOLD
        ):
            fail_text = "声纹注册失败，需大于三秒哦~"
            logger.log(
                "enroll_failed_too_short",
                duration_s=round(enroll_accum_seconds, 2),
                message=fail_text
            )
            print(fail_text)
            system_introduction(fail_text)
            flag_sv_enroll = 0
            enroll_started_ts = 0.0
            enroll_accum_seconds = 0.0
            enroll_accum_frames = []
            recent_kws_pinyin = []

        data = stream.read(CHUNK, exception_on_overflow=False)
        if time.time() < tts_busy_until:
            audio_buffer.clear()
            segments_to_save.clear()
            continue
        audio_buffer.append(data)
        
        # 按窗口检测一次 VAD
        if len(audio_buffer) * CHUNK / AUDIO_RATE >= VAD_WINDOW_SECONDS:
            # 拼接音频数据并检测 VAD
            raw_audio = b''.join(audio_buffer)
            vad_result = check_vad_activity(raw_audio)
            
            if vad_result:
                last_active_time = time.time()
                segments_to_save.append((raw_audio, time.time()))
            
            audio_buffer = []  # 清空缓冲区
        
        # 检查无效语音时间
        if time.time() - last_active_time > NO_SPEECH_THRESHOLD:
            # 检查是否需要保存
            if segments_to_save and segments_to_save[-1][1] > last_vad_end_time:
                seg_duration = VAD_WINDOW_SECONDS * len(segments_to_save)
                if seg_duration >= MIN_UTTERANCE_SECONDS:
                    save_audio_video()
                else:
                    logger.log("segment_dropped_short", duration_s=round(seg_duration, 2))
                    segments_to_save.clear()
                last_active_time = time.time()
            else:
                pass
                # print("无新增语音段，跳过保存")
    
    stream.stop_stream()
    stream.close()
    p.terminate()

# 视频录制线程
def video_recorder():
    global video_queue, recording_active
    
    cap = cv2.VideoCapture(0)  # 使用默认摄像头
    print("视频录制已开始")
    
    while recording_active:
        ret, frame = cap.read()
        if ret:
            video_queue.put((frame, time.time()))
            
            # 实时显示摄像头画面
            cv2.imshow("Real Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 Q 键退出
                break
        else:
            print("无法获取摄像头画面")
    
    cap.release()
    cv2.destroyAllWindows()

# 检测 VAD 活动
def check_vad_activity(audio_data):
    # 将音频数据分块检测
    num, rate = 0, 0.5
    bytes_per_sample = 2 * AUDIO_CHANNELS
    step = int(AUDIO_RATE * 0.02) * bytes_per_sample
    flag_rate = round(rate * (len(audio_data) // step))

    for i in range(0, len(audio_data), step):
        chunk = audio_data[i:i + step]
        if len(chunk) == step:
            if vad.is_speech(chunk, sample_rate=AUDIO_RATE):
                num += 1

    if num > flag_rate:
        return True
    return False

# 保存音频和视频
def save_audio_video():
    global segments_to_save, video_queue, last_vad_end_time, saved_intervals

    # 全局变量，用于保存音频文件名计数
    global audio_file_count
    global flag_sv_enroll
    global set_SV_enroll
    global enroll_accum_frames, enroll_accum_seconds, enroll_started_ts

    if flag_sv_enroll:
        audio_output_path = os.path.join(set_SV_enroll, "enroll_0.wav")
    else:
        audio_file_count = (audio_file_count % MAX_FILES) + 1
        audio_output_path = os.path.join(OUTPUT_DIR, f"audio_{audio_file_count}.wav")
    # audio_output_path = f"{OUTPUT_DIR}/audio_0.wav"

    if not segments_to_save:
        return
    
    # 停止当前播放的音频
    if stop_audio_playback():
        print("检测到新的有效音，已停止当前音频播放")
        
    # 获取有效段的时间范围
    start_time = segments_to_save[0][1]
    end_time = segments_to_save[-1][1]
    
    # 检查是否与之前的片段重叠
    if saved_intervals and saved_intervals[-1][1] >= start_time:
        print("当前片段与之前片段重叠，跳过保存")
        segments_to_save.clear()
        return
    
    # 保存音频
    audio_frames = [seg[0] for seg in segments_to_save]

    if flag_sv_enroll:
        audio_length = VAD_WINDOW_SECONDS * len(segments_to_save)
        enroll_accum_frames.extend(audio_frames)
        enroll_accum_seconds += audio_length
        logger.log(
            "enroll_collecting",
            duration_s=round(enroll_accum_seconds, 2),
            message=f"声纹注册累计时长 {enroll_accum_seconds:.1f}s / {ENROLL_MIN_SECONDS:.1f}s"
        )
        segments_to_save.clear()

        if enroll_accum_seconds < ENROLL_MIN_SECONDS:
            return

        os.makedirs(os.path.dirname(os.path.abspath(audio_output_path)), exist_ok=True)
        wf = wave.open(audio_output_path, 'wb')
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b''.join(enroll_accum_frames))
        wf.close()
        print(f"音频保存至 {audio_output_path}")
        enroll_accum_frames = []
        enroll_accum_seconds = 0.0
        enroll_started_ts = 0.0
    else:
        os.makedirs(os.path.dirname(os.path.abspath(audio_output_path)), exist_ok=True)
        wf = wave.open(audio_output_path, 'wb')
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b''.join(audio_frames))
        wf.close()
        print(f"音频保存至 {audio_output_path}")

    # Inference()

    if flag_sv_enroll:
        text = "声纹注册完成！现在只有你可以命令我啦！"
        print(text)
        flag_sv_enroll = 0
        system_introduction(text)
    else:
        # 同时只允许一个推理任务，避免堆积导致时延持续升高
        global inference_running
        with inference_lock:
            if inference_running:
                logger.log("inference_busy_drop", text=audio_output_path)
                segments_to_save.clear()
                return
            inference_running = True

        inference_thread = threading.Thread(target=inference_worker, args=(audio_output_path,), daemon=True)
        inference_thread.start()
        
        # 记录保存的区间
        saved_intervals.append((start_time, end_time))
        
    # 清空缓冲区
    segments_to_save.clear()

# --- 播放音频 -
def next_tts_output_path(prefix: str) -> str:
    global tts_file_seq, folder_path
    ts = time.strftime("%Y%m%d_%H%M%S")
    with tts_file_seq_lock:
        tts_file_seq += 1
        seq = tts_file_seq
    return os.path.join(folder_path, f"{prefix}_{ts}_{seq}.mp3")

def stop_audio_playback():
    with audio_playback_lock:
        stopped = False
        try:
            if pygame.mixer.get_init():
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    stopped = True
                pygame.mixer.quit()
        except Exception:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
        return stopped

def play_audio(file_path):
    with audio_playback_lock:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            print("播放完成！")
        except Exception as e:
            print(f"播放失败: {e}")
        finally:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            try:
                pygame.mixer.quit()
            except Exception:
                pass

async def amain(TEXT, VOICE, OUTPUT_FILE) -> None:
    """Main function"""
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)

import os

def is_folder_empty(folder_path):
    """
    检测指定文件夹内是否有文件。
    
    :param folder_path: 文件夹路径
    :return: 如果文件夹为空返回 True，否则返回 False
    """
    # 获取文件夹中的所有条目（文件或子文件夹）
    entries = os.listdir(folder_path)
    # 检查是否存在文件
    for entry in entries:
        # 获取完整路径
        full_path = os.path.join(folder_path, entry)
        # 如果是文件，返回 False
        if os.path.isfile(full_path):
            return False
    # 如果没有文件，返回 True
    return True


def sanitize_model_output(text):
    if not text:
        return text
    return re.sub(r'^\s*(系统|system|assistant)\s*[:：]\s*', '', text, flags=re.IGNORECASE).strip()


def normalize_text(text):
    text = text or ""
    return re.sub(r'[^\w\u4e00-\u9fa5]+', '', text.lower())


def strip_wakeword_prefix(text):
    t = (text or "").strip()
    # 去掉句首招呼词 + 唤醒词，减少误把唤醒词本身送入LLM导致慢响应
    t = re.sub(
        r'^\s*(你好|嗨|哈喽|hello|hi|hey|こんにちは|もしもし)?\s*[，,\s]*'
        r'(小[千倩钱]|晓倩|xiaoqian|xiao\s*qian)\s*[，,\s]*',
        '',
        t,
        flags=re.IGNORECASE
    )
    return t.strip()


def has_han(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text or ""))


def has_kana(text):
    return bool(re.search(r'[\u3040-\u30ff]', text or ""))


def latin_ratio(text):
    if not text:
        return 0.0
    letters = re.findall(r'[A-Za-z]', text)
    return len(letters) / max(1, len(text))


def detect_speaker_lang(reply_text, user_text=""):
    """
    语种判定优先级：
    1) 文本脚本特征（汉字/日文假名/拉丁字母）
    2) langid 兜底
    """
    text = (reply_text or "").strip()
    user = (user_text or "").strip()

    # 优先根据用户提问语种决定音色，避免“英文提问但中文音色”
    if has_kana(user):
        return "ja"
    # 混合文本（如 xiao倩ian）优先按拉丁占比判定为英文，避免误回中文音色
    if latin_ratio(user) > 0.45 and not has_kana(user):
        return "en"
    if has_han(user):
        return "zh"

    if has_kana(text):
        return "ja"
    if latin_ratio(text) > 0.55 and not has_kana(text):
        return "en"
    if has_han(text):
        return "zh"

    try:
        lang, _ = langid.classify(user or text)
        return lang
    except Exception:
        return "zh"


def fast_reply(prompt_text):
    """
    轻量快速回复：优先覆盖常见高频问句，减少LLM调用等待。
    """
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
    if ("你能回答" in t) or ("能回答我的问题" in t):
        return "当然可以，你直接问我就行。"
    if ("几个季节" in t):
        return "一年有四个季节：春夏秋冬。"
    if ("你知道我想让你干什么" in t):
        return "你直接说指令，我马上执行。"
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


def normalize_pinyin(text):
    text = (text or "").lower()
    # 唤醒词归一化：同时保留英文、日文假名与中日韩统一表意文字
    text = re.sub(r'[^a-z0-9\u3040-\u30ff\u4e00-\u9fff\s]+', '', text)
    return ' '.join(text.split())


def is_kws_triggered(pinyin_text):
    global recent_kws_pinyin, kws_aliases
    norm = normalize_pinyin(pinyin_text)
    if not norm:
        return False

    recent_kws_pinyin.append(norm)
    if len(recent_kws_pinyin) > 6:
        recent_kws_pinyin = recent_kws_pinyin[-6:]

    joined = " ".join(recent_kws_pinyin)
    joined_compact = joined.replace(" ", "")

    for alias in kws_aliases:
        alias_norm = normalize_pinyin(alias)
        alias_compact = alias_norm.replace(" ", "")
        if not alias_compact:
            continue
        if alias_norm in joined or alias_compact in joined_compact:
            recent_kws_pinyin = []
            return True
        tail = joined_compact[-(len(alias_compact) + 4):]
        if len(tail) >= max(4, len(alias_compact) - 3):
            if SequenceMatcher(None, tail, alias_compact).ratio() >= 0.82:
                recent_kws_pinyin = []
                return True
    return False


def build_kws_candidate(text):
    """
    唤醒词匹配输入：
    - 中文部分 -> 拼音
    - 英文部分 -> 原文归一化
    两者拼接后统一匹配，支持中英唤醒词。
    """
    py = extract_chinese_and_convert_to_pinyin(text)
    en = normalize_pinyin(text)
    if py and en:
        return f"{py} {en}"
    return py or en


def is_echo_text(asr_text):
    global last_tts_text, last_tts_ts
    if time.time() - last_tts_ts > 8:
        return False
    a = normalize_text(asr_text)
    b = normalize_text(last_tts_text)
    if len(a) < 4 or len(b) < 4:
        return False
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.85


def inference_worker(path):
    global inference_running
    try:
        Inference(path)
    finally:
        with inference_lock:
            inference_running = False


# -------- SenceVoice 语音识别 --模型加载-----
model_dir = r"D:\AI_Models\SenseVoiceSmall"
model_senceVoice = AutoModel(model=model_dir, trust_remote_code=False, disable_update=True)
# -------- CAM++声纹识别 -- 模型加载 --------
set_SV_enroll = os.path.join(BASE_DIR, "SpeakerVerification_DIR", "enroll_wav")
os.makedirs(set_SV_enroll, exist_ok=True)

def clear_sv_enroll_files():
    global enroll_accum_frames, enroll_accum_seconds, enroll_started_ts
    try:
        enroll_dir = os.path.abspath(set_SV_enroll)
        if os.path.isdir(enroll_dir):
            for name in os.listdir(enroll_dir):
                p = os.path.join(enroll_dir, name)
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
    except Exception:
        pass
    enroll_accum_frames = []
    enroll_accum_seconds = 0.0
    enroll_started_ts = 0.0

# 不在进程启动/退出时自动清空声纹文件，保证 GUI 问答与控制可复用同一注册信息。
sv_pipeline = pipeline(
    task='speaker-verification',
    model='damo/speech_campplus_sv_zh-cn_16k-common',
    model_revision='v1.0.0'
)

# --------- QWen2.5大语言模型 ---------------

model_name = r"D:\AI_Models\Qwen2.5-1.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype
)
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
try:
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
except Exception:
    pass
# ---------- 模型加载结束 -----------------------

class ChatMemory:
    def __init__(self, max_length=2048):
        self.history = []
        self.max_length = max_length

    def add_to_history(self, user_input, model_response):
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": model_response})
        self._truncate()

    def _truncate(self):
        def total_chars():
            return sum(len(x.get("content", "")) for x in self.history)

        while self.history and total_chars() > self.max_length:
            self.history.pop(0)

    def get_context(self):
        return list(self.history)
    
# -------- memory 初始化 --------
memory = ChatMemory(max_length=160)

def system_introduction(text):
    global folder_path, last_tts_text, last_tts_ts, tts_busy_until
    text = sanitize_model_output(text)
    print("LLM output:", text)
    logger.log("system_intro", text=text)
    used_speaker = "zh-CN-XiaoyiNeural"
    output_mp3 = next_tts_output_path("sft_tmp")
    stop_audio_playback()
    last_tts_text = text
    last_tts_ts = time.time()
    asyncio.run(amain(text, used_speaker, output_mp3))
    play_audio(output_mp3)
    tts_busy_until = time.time() + 0.45

def Inference(TEMP_AUDIO_FILE=f"{OUTPUT_DIR}/audio_0.wav"):
    '''
    1. 使用senceVoice做asr，转换为拼音，检测唤醒词
        - 首先检测声纹注册文件夹是否有注册文件，如果无，启动声纹注册
    2. 使用CAM++做声纹识别
        - 设置固定声纹注册语音目录，每次输入音频均进行声纹对比
    3. 以上两者均通过，则进行大模型推理
    '''
    global audio_file_count
    global set_SV_enroll
    global flag_sv_enroll
    global thred_sv
    global flag_sv_used
    global set_KWS
    global flag_KWS
    global flag_KWS_used
    global last_kws_hint_ts
    global enroll_accum_frames, enroll_accum_seconds, enroll_started_ts
    global last_tts_text, last_tts_ts, tts_busy_until

    os.makedirs(set_SV_enroll, exist_ok=True)

    # 1) ASR 先行
    t0 = time.perf_counter()
    res = model_senceVoice.generate(
        input=TEMP_AUDIO_FILE,
        cache={},
        language="auto",
        use_itn=False,
    )
    asr_ms = (time.perf_counter() - t0) * 1000
    prompt = res[0]["text"].split(">")[-1].strip()
    if not prompt:
        return

    if len(normalize_text(prompt)) < 2:
        logger.log("asr_too_short_drop", text=prompt)
        return

    if is_echo_text(prompt):
        logger.log("echo_filtered", text=prompt)
        return

    prompt_kws = build_kws_candidate(prompt)
    logger.log("asr_done", text=prompt, pinyin=prompt_kws, asr_ms=round(asr_ms, 2))

    enroll_wav = os.path.join(set_SV_enroll, "enroll_0.wav")
    has_enroll = os.path.exists(enroll_wav)

    # 2) 统一策略：只要没说唤醒词“小千”，直接忽略。
    kws_hit = is_kws_triggered(prompt_kws)
    must_kws = True if flag_KWS_used else False
    if must_kws and not kws_hit:
        if not has_enroll:
            logger.log("ignore_no_enroll_without_kws", text=prompt)
            if time.time() - last_kws_hint_ts > 4.0:
                print("等待唤醒词“小千”以开始声纹注册")
                last_kws_hint_ts = time.time()
        else:
            logger.log("ignore_without_kws", text=prompt)
        return

    # 3) 已唤醒但还没有注册声纹：直接进入注册流程（不再重复播报）
    if flag_sv_used and not has_enroll:
        print("检测到唤醒词，开始录入声纹，请连续说话大于三秒")
        logger.log("trigger_enroll_by_kws", text=prompt, message="start_enroll")
        enroll_accum_frames = []
        enroll_accum_seconds = 0.0
        enroll_started_ts = time.time()
        flag_sv_enroll = 1
        return

    # 4) 已唤醒 + 已注册：做SV门禁
    if flag_sv_used:
        sv_score = sv_pipeline([enroll_wav, TEMP_AUDIO_FILE], thr=thred_sv)
        logger.log("sv_result", message=f"sv={sv_score.get('text')}", result=sv_score, text=prompt)
        if sv_score.get("text") != "yes":
            logger.log("filtered_by_sv", text=prompt)
            system_introduction("声纹识别失败，不好意思我不能为你服务")
            return

    # 支持重新注册口令
    if ("重新注册" in prompt) or ("重新录入" in prompt) or ("声纹注册" in prompt):
        try:
            if os.path.exists(enroll_wav):
                os.remove(enroll_wav)
        except Exception:
            pass
        flag_sv_enroll = 1
        enroll_accum_frames = []
        enroll_accum_seconds = 0.0
        enroll_started_ts = time.time()
        text = "好的，请连续说三秒用于声纹注册"
        print(text)
        system_introduction(text)
        return

    # 5) LLM 问答（已通过 唤醒词+声纹）
    prompt_tmp = strip_wakeword_prefix(prompt)
    if not prompt_tmp:
        output_text = "我在，你说。"
        logger.log("fast_reply_hit", text="wake_only")
    else:
        history_messages = memory.get_context()[-1:]
        quick_answer = fast_reply(prompt_tmp)
        if quick_answer is not None:
            output_text = quick_answer
            logger.log("fast_reply_hit", text=prompt_tmp)
        else:
            user_lang = detect_speaker_lang("", prompt_tmp)
            lang_rule = ""
            if user_lang == "en":
                lang_rule = "请使用简洁英文回答。"
            elif user_lang == "ja":
                lang_rule = "日本語で簡潔に答えてください。"
            messages = [
                {"role": "system", "content": f"你叫小千，是一个18岁的女大学生，性格活泼开朗，说话俏皮简洁，回答问题不会超过50字。{lang_rule}"},
            ]
            messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt_tmp})
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            llm_t0 = time.perf_counter()
            with torch.inference_mode():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=24,
                    do_sample=False,
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            llm_ms = (time.perf_counter() - llm_t0) * 1000
            logger.log("llm_infer_ms", latency_ms=round(llm_ms, 2), text=prompt_tmp)

    output_text = sanitize_model_output(output_text)
    logger.log("llm_done", prompt=prompt_tmp, answer=output_text)
    memory.add_to_history(prompt_tmp, output_text)

    text = output_text
    language = detect_speaker_lang(text, prompt_tmp)
    language_speaker = {
        "ja": "ja-JP-NanamiNeural",
        "fr": "fr-FR-DeniseNeural",
        "es": "ca-ES-JoanaNeural",
        "de": "de-DE-KatjaNeural",
        "zh": "zh-CN-XiaoyiNeural",
        "en": "en-US-AnaNeural",
    }

    if language not in language_speaker.keys():
        used_speaker = "zh-CN-XiaoyiNeural"
    else:
        used_speaker = language_speaker[language]
        print("检测到语种：", language, "使用音色：", language_speaker[language])

    output_mp3 = next_tts_output_path("sft")
    stop_audio_playback()
    last_tts_text = text
    last_tts_ts = time.time()
    asyncio.run(amain(text, used_speaker, output_mp3))
    play_audio(output_mp3)
    tts_busy_until = time.time() + 0.45

# 主函数
if __name__ == "__main__":

    try:
        # 启动音视频录制线程
        audio_thread = threading.Thread(target=audio_recorder)
        # video_thread = threading.Thread(target=video_recorder)
        audio_thread.start()
        # video_thread.start()

        if flag_sv_used and flag_KWS_used:
            enroll_wav = os.path.join(set_SV_enroll, "enroll_0.wav")
            if os.path.exists(enroll_wav):
                text = "您已开启声纹识别和关键词唤醒，请说“小千”后再提问。"
            else:
                text = "您已开启声纹识别和关键词唤醒，目前无声纹注册文件！请先注册声纹，需大于三秒哦~"
            system_introduction(text)
        elif flag_KWS_used:
            system_introduction("您已开启关键词唤醒。")
        elif flag_sv_used:
            system_introduction("您已开启声纹识别。")

        print("按 Ctrl+C 停止录制")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("录制停止中...")
        recording_active = False
        audio_thread.join()
        # video_thread.join()
        print("录制已停止")
