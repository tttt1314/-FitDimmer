#!/usr/bin/env bash
set -e

# 靜音 MediaPipe / GLog C++ 內部 Clearcut 遙測日誌
export GLOG_minloglevel=3
export MEDIAPIPE_DISABLE_TELEMETRY=1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ ! -d ".venv_fitdimmer" ]; then
    echo "[+] Creating virtual environment..."
    python3 -m venv .venv_fitdimmer
    ./.venv_fitdimmer/bin/pip install -r requirements.txt
fi

echo "=================================================="
echo "🏋️‍♂️ FitDimmer - macOS 肩膀動態感應螢幕調暗 App"
echo "=================================================="
echo "正在啟動 Web 儀表板..."
echo "網址: http://localhost:8000"
echo "按 Ctrl+C 可隨時關閉程式並復原原螢幕亮度"
echo "=================================================="

./.venv_fitdimmer/bin/python app.py
