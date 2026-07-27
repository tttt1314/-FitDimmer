import cv2
import numpy as np
import time
import os
import urllib.request
from collections import deque
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28)
]

class ShoulderMotionDetector:
    def __init__(self, idle_timeout=2.5, sensitivity=0.020):
        """
        :param idle_timeout: 秒數，當肩膀靜止超過此時間，狀態轉為 PAUSED 並調暗螢幕
        :param sensitivity: 平衡靈敏度門檻 (預設 0.020，濾除鏡頭噪點並抓取真實動作)
        """
        self.idle_timeout = idle_timeout
        self.sensitivity = sensitivity
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, 'pose_landmarker.task')
        
        if not os.path.exists(self.model_path):
            print(f"[PoseDetector] 下載 MediaPipe Pose 模型至: {self.model_path}...")
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
            urllib.request.urlretrieve(url, self.model_path)

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.IMAGE
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        
        # 歷史軌跡 Buffer (45 幀 ~ 1.5 秒)
        self.y_history = deque(maxlen=45)
        self.smoothed_y = None  # 指數平滑 EMA
        
        self.last_motion_time = time.time()
        self.is_active = True
        self.current_y_range = 0.0
        self.direction_changes = 0

    def set_parameters(self, idle_timeout=None, sensitivity=None):
        if idle_timeout is not None:
            self.idle_timeout = float(idle_timeout)
        if sensitivity is not None:
            self.sensitivity = float(sensitivity)

    def process_frame(self, frame):
        """
        處理單張影像：採用 EMA 低通濾波器抑制鏡頭噪點 + 轉折點 (Peaks & Troughs) 規律移動判斷
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        now = time.time()
        motion_detected = False
        shoulder_y = None
        
        results = self.landmarker.detect(mp_image)
        
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks = results.pose_landmarks[0]
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            
            if (0 <= l_shoulder.x <= 1) and (0 <= r_shoulder.x <= 1):
                raw_y = (l_shoulder.y + r_shoulder.y) / 2.0
                
                # 採用 EMA (Exponential Moving Average, alpha=0.35) 濾除攝影機鏡頭的高頻微小噪點
                if self.smoothed_y is None:
                    self.smoothed_y = raw_y
                else:
                    self.smoothed_y = 0.35 * raw_y + 0.65 * self.smoothed_y
                    
                shoulder_y = self.smoothed_y
                self.y_history.append(shoulder_y)
                
                # 分析滑動窗口內的肩膀上下擺動振幅與方向轉折點
                if len(self.y_history) >= 8:
                    y_arr = np.array(self.y_history)
                    self.current_y_range = float(np.max(y_arr) - np.min(y_arr))
                    
                    # 微分計算轉折點 (Peaks & Troughs)
                    dy = np.diff(y_arr)
                    # 忽略極微小的微速噪音 (thresholding noise)
                    valid_dy = np.where(np.abs(dy) > 0.0008, dy, 0.0)
                    
                    if len(valid_dy) > 2:
                        sign_changes = np.diff(np.sign(valid_dy))
                        self.direction_changes = int(np.count_nonzero(sign_changes != 0))
                    else:
                        self.direction_changes = 0

                    # 真正有效的運動判定：
                    # 振幅高於門檻，且存在方向轉折（上下來回擺動）；或位移振幅明顯大於 1.3 倍門檻
                    if (self.current_y_range >= self.sensitivity and self.direction_changes >= 1) or (self.current_y_range >= self.sensitivity * 1.35):
                        motion_detected = True
                        self.last_motion_time = now

                # 畫面繪製骨架與肩膀連線
                l_px = (int(l_shoulder.x * w), int(l_shoulder.y * h))
                r_px = (int(r_shoulder.x * w), int(r_shoulder.y * h))
                color = (0, 255, 0) if self.is_active else (0, 0, 235)
                
                for conn in POSE_CONNECTIONS:
                    p1 = (int(landmarks[conn[0]].x * w), int(landmarks[conn[0]].y * h))
                    p2 = (int(landmarks[conn[1]].x * w), int(landmarks[conn[1]].y * h))
                    cv2.line(frame, p1, p2, (180, 180, 180), 2)
                    
                cv2.line(frame, l_px, r_px, color, 4)
                cv2.circle(frame, l_px, 8, (255, 255, 255), -1)
                cv2.circle(frame, r_px, 8, (255, 255, 255), -1)
                cv2.circle(frame, l_px, 6, color, -1)
                cv2.circle(frame, r_px, 6, color, -1)

        # 計算靜止時間
        idle_duration = now - self.last_motion_time
        if idle_duration > self.idle_timeout:
            self.is_active = False
        else:
            self.is_active = True

        # HUD 畫面視覺標示
        status_str = "ACTIVE (Shoulder Moving)" if self.is_active else f"PAUSED (Idle: {idle_duration:.1f}s)"
        status_color = (0, 230, 0) if self.is_active else (0, 0, 235)
        
        cv2.rectangle(frame, (10, 10), (340, 105), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (340, 105), (60, 60, 60), 1)
        
        cv2.putText(frame, status_str, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"Y-Amp: {self.current_y_range:.4f} / Thresh: {self.sensitivity:.4f}", 
                    (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        
        # 動能進度條
        bar_w = int(min(300, (self.current_y_range / (self.sensitivity * 2.0)) * 300))
        cv2.rectangle(frame, (20, 78), (320, 92), (50, 50, 50), -1)
        if bar_w > 0:
            cv2.rectangle(frame, (20, 78), (20 + bar_w, 92), status_color, -1)

        info = {
            "is_active": self.is_active,
            "idle_duration": round(idle_duration if not self.is_active else 0.0, 2),
            "y_range": round(self.current_y_range, 4),
            "direction_changes": self.direction_changes,
            "shoulder_detected": shoulder_y is not None
        }
        
        return frame, info

    def close(self):
        self.landmarker.close()
