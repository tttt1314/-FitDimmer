import cv2
import numpy as np
import time
import os
import urllib.request
from collections import deque
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

# 定義 MediaPipe Pose 骨架連線
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28)
]

class ShoulderMotionDetector:
    def __init__(self, idle_timeout=2.5, sensitivity=0.03):
        """
        :param idle_timeout: 秒數，當肩膀靜止超過此時間，狀態轉為 PAUSED 並開始調暗螢幕
        :param sensitivity: 肩膀上下擺動靈敏度 (Y軸位移區間 0.01 ~ 0.08)
        """
        self.idle_timeout = idle_timeout
        self.sensitivity = sensitivity
        
        # 確定模型路徑
        model_filename = 'pose_landmarker.task'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, model_filename)
        
        if not os.path.exists(self.model_path):
            print(f"[PoseDetector] 下載 MediaPipe Pose 模型至: {self.model_path}...")
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
            urllib.request.urlretrieve(url, self.model_path)
            print("[PoseDetector] 模型下載完成！")

        # 初始化 MediaPipe PoseLandmarker
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.IMAGE
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        
        # 歷史軌跡 Buffer (60 幀 ~ 2 秒)
        self.y_history = deque(maxlen=60)
        self.last_motion_time = time.time()
        self.is_active = True
        self.current_y_range = 0.0
        self.current_speed = 0.0
        self.direction_changes = 0

    def set_parameters(self, idle_timeout=None, sensitivity=None):
        if idle_timeout is not None:
            self.idle_timeout = float(idle_timeout)
        if sensitivity is not None:
            self.sensitivity = float(sensitivity)

    def process_frame(self, frame):
        """
        處理單張影像，精準追蹤肩膀 (Keypoint 11 & 12) 上下移動
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
            
            # Keypoint 11: 左肩, Keypoint 12: 右肩
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]
            
            # 若肩膀點在畫面內
            if (0 <= l_shoulder.x <= 1) and (0 <= r_shoulder.x <= 1):
                shoulder_y = (l_shoulder.y + r_shoulder.y) / 2.0
                self.y_history.append(shoulder_y)
                
                # 分析最近 N 幀的肩膀 Y 軸上下移動軌跡
                if len(self.y_history) >= 10:
                    y_arr = np.array(self.y_history)
                    self.current_y_range = float(np.max(y_arr) - np.min(y_arr))
                    
                    # 計算垂直微分 (速度)
                    dy = np.diff(y_arr)
                    self.current_speed = float(np.mean(np.abs(dy))) if len(dy) > 0 else 0.0
                    
                    # 算方向轉折點 (Peaks & Troughs - 上下律動)
                    if len(dy) > 2:
                        sign_changes = np.diff(np.sign(dy))
                        self.direction_changes = int(np.count_nonzero(sign_changes != 0))
                    else:
                        self.direction_changes = 0

                    # 當 Y 軸高度移動區間大於門檻且有上下波動 -> 判定為「肩膀持續移動中」
                    if self.current_y_range >= self.sensitivity or self.current_speed >= self.sensitivity * 0.4:
                        motion_detected = True
                        self.last_motion_time = now

                # 繪製骨架與肩膀連線
                l_px = (int(l_shoulder.x * w), int(l_shoulder.y * h))
                r_px = (int(r_shoulder.x * w), int(r_shoulder.y * h))
                
                color = (0, 255, 0) if self.is_active else (0, 0, 235)
                
                # 繪製骨架圖層
                for conn in POSE_CONNECTIONS:
                    pt1 = landmarks[conn[0]]
                    pt2 = landmarks[conn[1]]
                    p1 = (int(pt1.x * w), int(pt1.y * h))
                    p2 = (int(pt2.x * w), int(pt2.y * h))
                    cv2.line(frame, p1, p2, (180, 180, 180), 2)
                    
                # 高亮肩膀連線
                cv2.line(frame, l_px, r_px, color, 4)
                cv2.circle(frame, l_px, 9, (255, 255, 255), -1)
                cv2.circle(frame, r_px, 9, (255, 255, 255), -1)
                cv2.circle(frame, l_px, 7, color, -1)
                cv2.circle(frame, r_px, 7, color, -1)

        # 計算靜止時間
        idle_duration = now - self.last_motion_time
        if idle_duration > self.idle_timeout:
            self.is_active = False
        else:
            self.is_active = True

        # HUD 畫面視覺標示
        status_str = "ACTIVE (Shoulder Moving)" if self.is_active else f"PAUSED (Idle: {idle_duration:.1f}s)"
        status_color = (0, 230, 0) if self.is_active else (0, 0, 235)
        
        cv2.rectangle(frame, (10, 10), (330, 105), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (330, 105), (60, 60, 60), 1)
        
        cv2.putText(frame, status_str, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"Y-Amp: {self.current_y_range:.4f} / Thresh: {self.sensitivity:.4f}", 
                    (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        
        # 動能進度條
        bar_w = int(min(290, (self.current_y_range / (self.sensitivity * 2.0)) * 290))
        cv2.rectangle(frame, (20, 78), (310, 92), (50, 50, 50), -1)
        if bar_w > 0:
            cv2.rectangle(frame, (20, 78), (20 + bar_w, 92), status_color, -1)

        info = {
            "is_active": self.is_active,
            "idle_duration": round(idle_duration if not self.is_active else 0.0, 2),
            "y_range": round(self.current_y_range, 4),
            "speed": round(self.current_speed, 4),
            "direction_changes": self.direction_changes,
            "shoulder_detected": shoulder_y is not None
        }
        
        return frame, info

    def close(self):
        self.landmarker.close()
