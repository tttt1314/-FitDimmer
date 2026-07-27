import os
os.environ["GLOG_minloglevel"] = "3"

import cv2
import time
import asyncio
import threading
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from pose_detector import ShoulderMotionDetector
from brightness_controller import BrightnessController

app = FastAPI(title="FitDimmer")
templates = Jinja2Templates(directory="templates")

# 全域控制器
brightness_ctrl = BrightnessController()
detector = ShoulderMotionDetector(idle_timeout=2.5, sensitivity=0.020)

# 設定參數
class SettingsModel(BaseModel):
    idle_timeout: float = 2.5
    sensitivity: float = 0.020
    dim_duration: float = 2.0
    min_brightness: float = 0.05

settings = SettingsModel()

# 狀態變數
latest_frame_bytes = None
frame_lock = threading.Lock()
prev_active_state = True
active_websockets: Set[WebSocket] = set()

def camera_loop():
    global latest_frame_bytes, prev_active_state
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] 無法開啟 webcam 鏡頭！請確認攝影機權限與連接狀態。")
        return

    print("[Info] Camera loop 啟動成功，開始監測肩膀動態...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        # 鏡頭畫面左右翻轉（鏡像）
        frame = cv2.flip(frame, 1)
        
        # 更新偵測器參數
        detector.set_parameters(
            idle_timeout=settings.idle_timeout,
            sensitivity=settings.sensitivity
        )
        
        # 處理幀姿態與動能
        processed_frame, info = detector.process_frame(frame)
        is_active = info["is_active"]
        
        # 狀態轉折判定 (Active <-> Paused)
        if prev_active_state and not is_active:
            print(f"[FitDimmer] 偵測到肩膀靜止 > {settings.idle_timeout}s，開始調暗螢幕至 {settings.min_brightness*100:.0f}%...")
            brightness_ctrl.fade_to(target_level=settings.min_brightness, duration=settings.dim_duration)
        elif not prev_active_state and is_active:
            print("[FitDimmer] 偵測到肩膀重新移動，即刻復原螢幕亮度！")
            brightness_ctrl.restore_brightness()
            
        prev_active_state = is_active

        # 編碼為 JPEG
        _, jpeg = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        with frame_lock:
            latest_frame_bytes = jpeg.tobytes()

        time.sleep(0.02) # ~30 fps

# 啟動相機背景執行緒
cam_thread = threading.Thread(target=camera_loop, daemon=True)
cam_thread.start()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

def generate_video_stream():
    global latest_frame_bytes
    while True:
        with frame_lock:
            if latest_frame_bytes is None:
                frame_data = b''
            else:
                frame_data = latest_frame_bytes
        
        if frame_data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.033)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/api/settings")
async def update_settings(new_settings: SettingsModel):
    global settings
    settings = new_settings
    detector.set_parameters(
        idle_timeout=settings.idle_timeout,
        sensitivity=settings.sensitivity
    )
    return {"status": "ok", "settings": settings.dict()}

@app.post("/api/restore")
async def restore_brightness_api():
    brightness_ctrl.restore_brightness()
    return {"status": "ok", "brightness": brightness_ctrl.get_brightness()}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            # 廣播即時監控數據
            curr_b = brightness_ctrl.current_brightness
            telemetry = {
                "is_active": detector.is_active,
                "idle_duration": round(time.time() - detector.last_motion_time if not detector.is_active else 0.0, 2),
                "idle_timeout": settings.idle_timeout,
                "y_range": round(detector.current_y_range, 4),
                "sensitivity": settings.sensitivity,
                "current_brightness": round(curr_b, 2)
            }
            await websocket.send_json(telemetry)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    print("[FitDimmer] 啟動伺服器：http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
