import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch

import sys
import numpy as np
import time
import re
import html
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QLabel, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QProcess

from voice_engine import VoiceEngine
from command_parser import CommandParser
from runtime_logger import RuntimeLogger

import pyqtgraph.opengl as gl

# --- 仿真逻辑：小车状态管理 ---
class CarState:
    def __init__(self):
        self.x, self.y = 0.0, 0.0
        self.yaw = 0.0  # 朝向（角度）
        self.history = [[0.0, 0.0, 0.0]] # 轨迹点记录
        self.actions = [] # 动作记录，用于回放 [(direction, distance), ...]
        self.last_action = None
        self.wheel_spin_velocity_deg = 0.0
        self.wheel_spin_until = 0.0
        self.turn_signal = None
        self.turn_signal_until = 0.0

    def move(self, direction, distance=1.0):
        """根据方向和距离更新坐标 (1步=1米)"""
        now = time.time()
        self.last_action = direction

        if direction == "forward":
            rad = np.radians(self.yaw)
            self.x += distance * np.sin(rad)
            self.y += distance * np.cos(rad)
            self.wheel_spin_velocity_deg = 520.0
            self.wheel_spin_until = now + max(0.35, min(1.4, 0.28 * max(distance, 1.0)))
        elif direction == "backward":
            rad = np.radians(self.yaw)
            self.x -= distance * np.sin(rad)
            self.y -= distance * np.cos(rad)
            self.wheel_spin_velocity_deg = -520.0
            self.wheel_spin_until = now + max(0.35, min(1.4, 0.28 * max(distance, 1.0)))
        elif direction == "left":
            self.yaw -= 90
            self.turn_signal = "left"
            self.turn_signal_until = now + 1.2
        elif direction == "right":
            self.yaw += 90
            self.turn_signal = "right"
            self.turn_signal_until = now + 1.2
        elif direction == "stop":
            self.wheel_spin_velocity_deg = 0.0
            self.wheel_spin_until = now
            self.turn_signal = None
            self.turn_signal_until = now
        
        self.history.append([self.x, self.y, 0.0])
        self.actions.append((direction, distance))
        return f"当前位置: ({self.x:.1f}, {self.y:.1f}), 朝向: {self.yaw % 360}°"

    def reset(self):
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0
        self.history = [[0.0, 0.0, 0.0]]
        self.actions = []
        self.last_action = None
        self.wheel_spin_velocity_deg = 0.0
        self.wheel_spin_until = 0.0
        self.turn_signal = None
        self.turn_signal_until = 0.0

# --- 3D 视图组件 ---
class RobotVisualizer(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setCameraPosition(distance=35, elevation=35, azimuth=45)

        grid = gl.GLGridItem()
        grid.setSize(40, 40)
        grid.setSpacing(1, 1)
        self.addItem(grid)

        axis_x = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [3, 0, 0]]), color=(1, 0, 0, 1), width=2, antialias=True)
        axis_y = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 3, 0]]), color=(0, 1, 0, 1), width=2, antialias=True)
        axis_z = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 2]]), color=(0.2, 0.6, 1, 1), width=2, antialias=True)
        self.addItem(axis_x)
        self.addItem(axis_y)
        self.addItem(axis_z)

        self.path_item = gl.GLLinePlotItem(pos=np.array([[0, 0, 0]], dtype=float), color=(0, 1, 1, 1), width=2, antialias=True)
        self.addItem(self.path_item)

        self.heading_item = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0.25], [0, 1.0, 0.25]], dtype=float),
            color=(1.0, 0.85, 0.15, 1.0),
            width=3,
            antialias=True
        )
        self.addItem(self.heading_item)

        self._build_car_body()
        self._build_wheels_and_indicators()

        self._state = None
        self._pose = (0.0, 0.0, 0.0)
        self._wheel_spin_angle_deg = 0.0
        self._last_anim_ts = time.perf_counter()

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._on_anim_tick)
        self.anim_timer.start(33)

    def _build_car_body(self):
        body_v, body_f = self._make_box_mesh(
            x_min=-0.42, x_max=0.42,
            y_min=-0.70, y_max=0.70,
            z_min=0.00, z_max=0.28
        )
        top_v, top_f = self._make_box_mesh(
            x_min=-0.30, x_max=0.30,
            y_min=-0.10, y_max=0.45,
            z_min=0.28, z_max=0.55
        )
        top_f = top_f + len(body_v)

        v = np.vstack([body_v, top_v]).astype(float)
        f = np.vstack([body_f, top_f]).astype(np.int32)
        mesh_data = gl.MeshData(vertexes=v, faces=f)
        self.car_body_item = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=False,
            color=(0.88, 0.16, 0.16, 1.0),
            shader="shaded",
            drawFaces=True,
            drawEdges=False
        )
        self.car_body_item.setGLOptions("opaque")
        self.addItem(self.car_body_item)

    def _build_wheels_and_indicators(self):
        self.wheel_radius = 0.20
        self.wheel_half_track = 0.48
        self.wheel_front_y = 0.52
        self.wheel_rear_y = -0.52
        self.wheel_z = 0.20

        self._wheel_centers_local = np.array([
            [-self.wheel_half_track, self.wheel_front_y, self.wheel_z],  # FL
            [ self.wheel_half_track, self.wheel_front_y, self.wheel_z],  # FR
            [-self.wheel_half_track, self.wheel_rear_y,  self.wheel_z],  # RL
            [ self.wheel_half_track, self.wheel_rear_y,  self.wheel_z],  # RR
        ], dtype=float)

        self._wheel_circle_template = self._make_wheel_circle_template(self.wheel_radius, points=36)
        self.wheel_ring_items = []
        self.wheel_spoke_items = []

        for _ in range(4):
            ring = gl.GLLinePlotItem(pos=np.zeros((36, 3), dtype=float), color=(0.08, 0.08, 0.08, 1.0), width=3, antialias=True)
            spoke = gl.GLLinePlotItem(pos=np.zeros((2, 3), dtype=float), color=(0.95, 0.95, 0.95, 1.0), width=2, antialias=True)
            self.wheel_ring_items.append(ring)
            self.wheel_spoke_items.append(spoke)
            self.addItem(ring)
            self.addItem(spoke)

        self.turn_indicator_local = np.array([
            [-0.26, 0.72, 0.40],  # left
            [ 0.26, 0.72, 0.40],  # right
        ], dtype=float)
        self.turn_indicator_item = gl.GLScatterPlotItem(
            pos=np.zeros((2, 3), dtype=float),
            size=14,
            color=np.array([[0.25, 0.25, 0.25, 0.65], [0.25, 0.25, 0.25, 0.65]], dtype=float),
            pxMode=True
        )
        self.turn_indicator_item.setGLOptions("opaque")
        self.addItem(self.turn_indicator_item)

    def update_pose(self, state: CarState):
        self._state = state
        self._pose = (state.x, state.y, state.yaw)
        self._apply_body_pose(state.x, state.y, state.yaw)
        self._update_heading(state.x, state.y, state.yaw)
        self._refresh_dynamic_parts()

        if len(state.history) > 1:
            self.path_item.setData(pos=np.array(state.history, dtype=float))

    def _on_anim_tick(self):
        now_perf = time.perf_counter()
        dt = max(0.0, min(0.2, now_perf - self._last_anim_ts))
        self._last_anim_ts = now_perf

        if self._state is not None and time.time() < self._state.wheel_spin_until:
            self._wheel_spin_angle_deg = (self._wheel_spin_angle_deg + self._state.wheel_spin_velocity_deg * dt) % 360.0

        self._refresh_dynamic_parts()

    def _apply_body_pose(self, x, y, yaw):
        self.car_body_item.resetTransform()
        self.car_body_item.rotate((-yaw) % 360.0, 0, 0, 1, local=False)
        self.car_body_item.translate(x, y, 0.0)

    def _update_heading(self, x, y, yaw):
        rad = np.radians(yaw)
        tip = np.array([x + 1.15 * np.sin(rad), y + 1.15 * np.cos(rad), 0.25], dtype=float)
        self.heading_item.setData(pos=np.array([[x, y, 0.25], tip], dtype=float))

    def _refresh_dynamic_parts(self):
        x, y, yaw = self._pose

        # 轮圈和轮辐
        for i, center_local in enumerate(self._wheel_centers_local):
            ring_local = self._wheel_circle_template + center_local
            ring_world = self._transform_points(ring_local, x, y, yaw)
            self.wheel_ring_items[i].setData(pos=ring_world)

            phi = np.radians(self._wheel_spin_angle_deg)
            spoke_local = np.array([
                center_local + np.array([0.0, self.wheel_radius * np.cos(phi), self.wheel_radius * np.sin(phi)]),
                center_local + np.array([0.0, -self.wheel_radius * np.cos(phi), -self.wheel_radius * np.sin(phi)]),
            ], dtype=float)
            spoke_world = self._transform_points(spoke_local, x, y, yaw)
            self.wheel_spoke_items[i].setData(pos=spoke_world)

        # 转向高亮（左/右闪烁）
        ind_world = self._transform_points(self.turn_indicator_local, x, y, yaw)
        colors = np.array([[0.25, 0.25, 0.25, 0.65], [0.25, 0.25, 0.25, 0.65]], dtype=float)
        now = time.time()
        if self._state is not None and now < self._state.turn_signal_until:
            blink_on = int(now * 6) % 2 == 0
            if blink_on:
                if self._state.turn_signal == "left":
                    colors[0] = [1.0, 0.78, 0.15, 1.0]
                elif self._state.turn_signal == "right":
                    colors[1] = [1.0, 0.78, 0.15, 1.0]
        self.turn_indicator_item.setData(pos=ind_world, size=14, color=colors)

    def _transform_points(self, points_local, x, y, yaw):
        pts = np.asarray(points_local, dtype=float)
        rad = np.radians(yaw)
        c = np.cos(rad)
        s = np.sin(rad)
        out = np.zeros_like(pts)
        out[:, 0] = x + pts[:, 0] * c + pts[:, 1] * s
        out[:, 1] = y - pts[:, 0] * s + pts[:, 1] * c
        out[:, 2] = pts[:, 2]
        return out

    def _make_wheel_circle_template(self, radius, points=36):
        theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
        y = radius * np.cos(theta)
        z = radius * np.sin(theta)
        x = np.zeros_like(theta)
        return np.column_stack([x, y, z]).astype(float)

    def _make_box_mesh(self, x_min, x_max, y_min, y_max, z_min, z_max):
        v = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max],
        ], dtype=float)
        f = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # -y
            [1, 5, 6], [1, 6, 2],  # +x
            [2, 6, 7], [2, 7, 3],  # +y
            [3, 7, 4], [3, 4, 0],  # -x
        ], dtype=np.int32)
        return v, f

# --- 主窗口 ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("小千机器人控制中心 - 语音交互 3D 仿真")
        self.resize(1300, 850)
        
        # 核心组件
        self.car = CarState()
        self.parser = CommandParser()
        self.engine = VoiceEngine()
        self.logger = RuntimeLogger()
        self.qna_process = QProcess(self)
        self.qna_process.setProcessChannelMode(QProcess.MergedChannels)
        self.qna_process.readyReadStandardOutput.connect(self.on_qna_output)
        self.qna_stdout_buffer = ""
        self.qna_last_line = ""
        self.last_status_msg = ""
        self.last_status_ts = 0.0
        self.last_runtime_line = ""
        self.last_runtime_ts = 0.0
        self._pending_mode_after_enroll = None
        self.clear_enroll_files()
        self.logger.log("app_init", message="MainWindow initialized")
        
        # 回放计时器
        self.replay_timer = QTimer()
        self.replay_timer.timeout.connect(self.do_replay_step)
        self.replay_queue = []
        self.motion_timer = QTimer()
        self.motion_timer.timeout.connect(self.do_continuous_motion_step)
        self.continuous_action = None

        self.init_ui()
        self.init_voice_engine()

    def clear_enroll_files(self):
        enroll_dir = os.path.abspath(self.engine.set_SV_enroll)
        try:
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

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- 左侧：3D 仿真与控制区 ---
        left_layout = QVBoxLayout()
        
        # 标题提示
        title_label = QLabel("🤖 3D 机器人运动仿真")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 5px;")
        left_layout.addWidget(title_label)

        self.visualizer = RobotVisualizer()
        left_layout.addWidget(self.visualizer, stretch=5)
        self.visualizer.update_pose(self.car)
        
        # 状态指示
        self.status_label = QLabel("系统初始化中...")
        self.status_label.setStyleSheet("font-size: 16px; color: #555; background: #f0f0f0; padding: 10px; border-radius: 5px;")
        left_layout.addWidget(self.status_label)

        # 按钮组
        btn_layout = QHBoxLayout()
        self.btn_clear = QPushButton("🗑️ 清空轨迹")
        self.btn_replay = QPushButton("🔄 轨迹回放")
        self.btn_voice_ctrl = QPushButton("🎤 语音控制模式 (已关闭)")
        
        # 样式美化
        for btn in [self.btn_clear, self.btn_replay, self.btn_voice_ctrl]:
            btn.setFixedHeight(50)
            btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.btn_voice_ctrl.setStyleSheet("background-color: #6c757d; color: white; height: 50px; font-weight: bold;")
        
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_replay)
        btn_layout.addWidget(self.btn_voice_ctrl, stretch=2)
        left_layout.addLayout(btn_layout)

        # --- 右侧：对话问答区 ---
        right_layout = QVBoxLayout()
        
        chat_title = QLabel("💬 小千智能问答")
        chat_title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 5px;")
        right_layout.addWidget(chat_title)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("font-size: 17px; line-height: 1.6; padding: 10px;")
        self.chat_display.setPlaceholderText("小千正在待命...")
        
        self.btn_chat = QPushButton("🗣️ 语音问答模式 (已关闭)")
        self.btn_chat.setFixedHeight(60)
        self.btn_chat.setStyleSheet("background-color: #6c757d; color: white; font-weight: bold; font-size: 16px;")
        
        right_layout.addWidget(self.chat_display)
        right_layout.addWidget(self.btn_chat)

        layout.addLayout(left_layout, stretch=3)
        layout.addLayout(right_layout, stretch=1)

        # 信号连接
        self.btn_clear.clicked.connect(self.clear_action)
        self.btn_replay.clicked.connect(self.start_replay)
        self.btn_voice_ctrl.clicked.connect(self.toggle_control_mode)
        self.btn_chat.clicked.connect(self.toggle_chat_mode)

    def init_voice_engine(self):
        """连接语音引擎信号"""
        self.engine.signals.status_update.connect(self.on_voice_status)
        self.engine.signals.asr_result.connect(self.on_asr_result)
        self.engine.signals.llm_result.connect(self.on_llm_result)
        self.engine.signals.control_text.connect(self.on_control_text)
        self.engine.signals.runtime_event.connect(self.on_engine_runtime_event)
        self.engine.signals.error_occurred.connect(self.on_voice_error)

    def ensure_engine_running(self):
        if not self.engine.isRunning():
            self.engine.start()
            self.logger.log("engine_start", message="VoiceEngine started")

    def pause_engine(self):
        if self.engine.isRunning():
            self.engine.mode = "none"
            self.engine.stop()
            self.logger.log("engine_stop", message="VoiceEngine stopped")

    # --- 槽函数 ---
    def _now_ts(self):
        now = time.time()
        base = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        ms = int((now - int(now)) * 1000)
        return f"{base}.{ms:03d}"

    def _append_runtime_line(self, line, color="#888"):
        now = time.time()
        # 简单去重：短时间内重复同一行不再刷屏
        if line == self.last_runtime_line and (now - self.last_runtime_ts) < 0.8:
            return
        self.last_runtime_line = line
        self.last_runtime_ts = now
        safe = html.escape(line)
        self.chat_display.append(
            f"<span style='color:{color}; font-family:Consolas, \"Courier New\", monospace; font-size:17px'>{safe}</span>"
        )
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def _append_runtime_event(self, event, text="", color="#888"):
        msg = f"[{self._now_ts()}] {event}"
        if text:
            msg = f"{msg} {text}"
        self._append_runtime_line(msg, color=color)

    def _clean_qna_line(self, line):
        # 15.1 子进程日志直通：仅去掉 ANSI 控制字符，不改业务文本
        line = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line or "")
        return line.strip()

    def on_voice_status(self, msg):
        self.status_label.setText(f"状态: {msg}")
        if self.qna_process.state() != QProcess.Running:
            now = time.time()
            # 这些事件会由 runtime_event 统一输出，避免重复
            skip_log_prefixes = ("sv_result ", "指令识别:")
            skip_log_equals = {"未识别到控制指令", "声纹识别失败，不好意思我不能为你服务"}
            if (
                not msg.startswith(skip_log_prefixes)
                and msg not in skip_log_equals
                and (msg != self.last_status_msg or (now - self.last_status_ts) > 1.5)
            ):
                self._append_runtime_event("status_update", msg, color="#666")
                self.last_status_msg = msg
                self.last_status_ts = now
        if "完成" in msg or "就绪" in msg:
            self.status_label.setStyleSheet("font-size: 16px; color: #28a745; background: #e9f7ef; padding: 10px;")
        if "声纹注册完成" in msg and self._pending_mode_after_enroll == "control":
            self._pending_mode_after_enroll = None
            self.engine.mode = "control"
            self.btn_voice_ctrl.setText("🎤 语音控制模式 (运行中)")
            self.btn_voice_ctrl.setStyleSheet("background-color: #28a745; color: white; height: 50px; font-weight: bold;")
            self.btn_chat.setText("🗣️ 语音问答模式 (已关闭)")
            self.btn_chat.setStyleSheet("background-color: #6c757d; color: white; font-weight: bold; font-size: 16px;")
            self.status_label.setText("控制模式：声纹已注册，可以开始下达指令")

    def on_asr_result(self, text):
        if self.qna_process.state() != QProcess.Running:
            self._append_runtime_event("asr_done", text, color="#333")

    def on_engine_runtime_event(self, event, text):
        if self.qna_process.state() == QProcess.Running:
            return
        color_map = {
            "sv_result": "#5a32a3",
            "sv_cache_hit": "#5a32a3",
            "filtered_by_sv": "#b00020",
            "control_stop_bypass_sv": "#b00020",
            "control_text_emit": "#0d4f8b",
            "control_trigger_enroll": "#9a6700",
            "control_non_command_drop": "#b36b00",
            "ignore_no_enroll_without_kws": "#666",
        }
        self._append_runtime_event(event, text, color=color_map.get(event, "#666"))

    def on_control_text(self, text):
        parse_t0 = time.perf_counter()
        commands = self.parser.parse_sequence(text)
        parse_ms = (time.perf_counter() - parse_t0) * 1000
        self.logger.log("control_asr_text", text=text, parse_ms=round(parse_ms, 2))
        if self.qna_process.state() != QProcess.Running:
            self._append_runtime_event("control_asr_text", text, color="#444")

        if not commands:
            self.status_label.setText(f"❓ 未能识别指令: {text}")
            self.logger.log("control_unrecognized", text=text)
            if self.qna_process.state() != QProcess.Running:
                self._append_runtime_event("control_unrecognized", text, color="#b36b00")
            return

        # 停止指令最高优先级：一旦出现立即生效，不再执行其余动作
        if any(cmd["action"] == "stop" for cmd in commands):
            self._stop_continuous_motion()
            self.status_label.setText("⏹ 已停止持续运动")
            self.logger.log("control_stop", text=text)
            if self.qna_process.state() != QProcess.Running:
                self._append_runtime_event("control_stop", text, color="#b00020")
            return

        # 冲突可打断：新指令到达时中断持续运动
        if self.continuous_action:
            self._stop_continuous_motion()
            self.logger.log("control_interrupt", message="continuous action interrupted by new command")

        executed = []
        for cmd in commands:
            action = cmd["action"]
            dist = cmd["distance"]
            continuous = cmd["continuous"]

            if action == "stop":
                self._stop_continuous_motion()
                executed.append("stop")
                continue

            if action in ("forward", "backward") and continuous:
                self.continuous_action = action
                # 立即执行一步，然后每秒继续执行一步
                info = self.car.move(action, 1.0)
                self.visualizer.update_pose(self.car)
                self.motion_timer.start(1000)
                executed.append(f"{action}(continuous)")
                self.logger.log("control_continuous_start", action=action, message=info)
                continue

            info = self.car.move(action, dist)
            self.visualizer.update_pose(self.car)
            executed.append(f"{action}:{dist}")
            self.logger.log("control_execute", action=action, distance=dist, message=info)

        self.status_label.setText(f"✅ 指令执行: {' | '.join(executed)}")
        if self.qna_process.state() != QProcess.Running:
            self._append_runtime_event("control_done", " | ".join(executed), color="#1d6f42")

    def _stop_continuous_motion(self):
        if self.motion_timer.isActive():
            self.motion_timer.stop()
        self.continuous_action = None
        self.logger.log("control_continuous_stop", message="continuous action stopped")

    def do_continuous_motion_step(self):
        if not self.continuous_action:
            self.motion_timer.stop()
            return
        info = self.car.move(self.continuous_action, 1.0)
        self.visualizer.update_pose(self.car)
        self.status_label.setText(f"▶ 持续执行: {self.continuous_action} | {info}")
        self.logger.log("control_continuous_step", action=self.continuous_action, message=info)

    def on_llm_result(self, text):
        if self.qna_process.state() != QProcess.Running:
            self._append_runtime_event("llm_done", text, color="#0056b3")

    def on_voice_error(self, err):
        QMessageBox.critical(self, "语音引擎错误", err)

    def toggle_control_mode(self):
        """切换控制模式"""
        if self.engine.mode != "control":
            # 控制模式与问答模式互斥，先停止 15.1 子进程避免抢麦
            self.stop_qna_process()
            self.ensure_engine_running()

            enroll_path = os.path.join(os.path.abspath(self.engine.set_SV_enroll), "enroll_0.wav")
            if not os.path.exists(enroll_path):
                self._pending_mode_after_enroll = "control"
                self.engine.mode = "control"
                self.status_label.setText("控制模式需先注册声纹：请先说“小千”触发注册，再连续说话≥3秒")
            else:
                self._pending_mode_after_enroll = None
                self.engine.mode = "control"
                self.status_label.setText("进入控制模式：直接说“前进/后退/左转/右转”等指令（先做声纹校验）")

            self.btn_voice_ctrl.setText("🎤 语音控制模式 (运行中)")
            self.btn_voice_ctrl.setStyleSheet("background-color: #28a745; color: white; height: 50px; font-weight: bold;")
            self.btn_chat.setText("🗣️ 语音问答模式 (已关闭)")
            self.btn_chat.setStyleSheet("background-color: #6c757d; color: white; font-weight: bold; font-size: 16px;")
        else:
            self.on_voice_status("待命")
            self.pause_engine()
            self.btn_voice_ctrl.setText("🎤 语音控制模式 (已关闭)")
            self.btn_voice_ctrl.setStyleSheet("background-color: #6c757d; color: white; height: 50px; font-weight: bold;")

    def toggle_chat_mode(self):
        """切换问答模式"""
        if self.qna_process.state() != QProcess.Running:
            # 问答模式与控制模式互斥，先暂停控制引擎避免抢麦
            self._stop_continuous_motion()
            self.pause_engine()
            self.start_qna_process()
            self.btn_chat.setText("🗣️ 语音问答模式 (运行中)")
            self.btn_chat.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; font-size: 16px;")
            self.btn_voice_ctrl.setText("🎤 语音控制模式 (已关闭)")
            self.btn_voice_ctrl.setStyleSheet("background-color: #6c757d; color: white; height: 50px; font-weight: bold;")
            self.status_label.setText("问答模式：由 15.1_SenceVoice_kws_CAM++.py 提供（含唤醒词/声纹/记忆）")
        else:
            self.stop_qna_process()
            self.btn_chat.setText("🗣️ 语音问答模式 (已关闭)")
            self.btn_chat.setStyleSheet("background-color: #6c757d; color: white; font-weight: bold; font-size: 16px;")
            self.status_label.setText("问答模式已关闭")

    def start_qna_process(self):
        script_path = os.path.join(os.path.dirname(__file__), "15.1_SenceVoice_kws_CAM++.py")
        self.qna_process.setWorkingDirectory(os.path.dirname(__file__))
        self.qna_stdout_buffer = ""
        self.qna_last_line = ""
        # 使用 -u 保持子进程实时输出，与终端单独运行保持一致
        self.qna_process.start(sys.executable, ["-u", script_path])
        self.logger.log("qna_process_start", script=script_path)

    def stop_qna_process(self):
        if self.qna_process.state() == QProcess.Running:
            self.qna_process.terminate()
            if not self.qna_process.waitForFinished(1500):
                self.qna_process.kill()
            self.logger.log("qna_process_stop", message="QnA process stopped")

    def on_qna_output(self):
        try:
            data = bytes(self.qna_process.readAllStandardOutput()).decode(errors="ignore")
            if not data:
                return
            self.qna_stdout_buffer += data.replace("\r", "\n")
            lines = self.qna_stdout_buffer.split("\n")
            self.qna_stdout_buffer = lines.pop() if lines else ""
            for raw in lines:
                line = self._clean_qna_line(raw)
                if not line:
                    continue
                if line == self.qna_last_line:
                    continue
                self.qna_last_line = line
                self._append_runtime_line(line, color="#888")
        except Exception:
            pass

    def clear_action(self):
        self._stop_continuous_motion()
        self.car.reset()
        self.visualizer.update_pose(self.car)
        self.status_label.setText("轨迹与记录已清空")
        self.chat_display.clear()
        self.logger.log("ui_clear", message="trajectory and chat cleared")

    def closeEvent(self, event):
        try:
            self._stop_continuous_motion()
            self.stop_qna_process()
        except Exception:
            pass
        try:
            self.engine.stop()
        except Exception:
            pass
        self.clear_enroll_files()
        super().closeEvent(event)

    def start_replay(self):
        """开始回放轨迹"""
        self._stop_continuous_motion()
        if not self.car.actions:
            self.status_label.setText("提示: 没有可回放的动作记录")
            return
        
        self.status_label.setText("🎬 正在回放历史动作...")
        self.replay_queue = self.car.actions.copy()
        self.car.reset()
        self.visualizer.update_pose(self.car)
        self.replay_timer.start(1000) # 每秒执行一步

    def do_replay_step(self):
        if not self.replay_queue:
            self.replay_timer.stop()
            self.status_label.setText("✅ 回放结束")
            return
        
        action, dist = self.replay_queue.pop(0)
        self.car.move(action, dist)
        self.visualizer.update_pose(self.car)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
