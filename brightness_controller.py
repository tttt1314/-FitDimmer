import ctypes
import ctypes.util
import time
import threading
import atexit
import signal
import sys

class BrightnessController:
    def __init__(self):
        self._lock = threading.Lock()
        self.ds = None
        self._init_api()
        
        self.original_brightness = self.get_brightness()
        if self.original_brightness is None or self.original_brightness < 0.1:
            self.original_brightness = 0.8  # Fallback safe default
            
        self.target_brightness = self.original_brightness
        self.current_brightness = self.original_brightness
        
        self._fade_thread = None
        self._stop_fade = threading.Event()
        
        # Register cleanup hooks
        atexit.register(self.restore_brightness)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _init_api(self):
        try:
            ds_path = '/System/Library/PrivateFrameworks/DisplayServices.framework/DisplayServices'
            self.ds = ctypes.CDLL(ds_path)
            
            self._set_b = self.ds.DisplayServicesSetBrightness
            self._set_b.argtypes = [ctypes.c_uint32, ctypes.c_float]
            self._set_b.restype = ctypes.c_int
            
            self._get_b = self.ds.DisplayServicesGetBrightness
            self._get_b.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
            self._get_b.restype = ctypes.c_int
        except Exception as e:
            print(f"[BrightnessController] Warning: Could not load DisplayServices API: {e}")
            self.ds = None

    def _signal_handler(self, sig, frame):
        self.restore_brightness()
        sys.exit(0)

    def get_brightness(self, display_id=1):
        if not self.ds:
            return 0.8
        try:
            b = ctypes.c_float()
            res = self._get_b(display_id, ctypes.byref(b))
            if res == 0:
                return float(b.value)
        except Exception as e:
            print(f"[BrightnessController] Error getting brightness: {e}")
        return 0.8

    def set_brightness(self, level, display_id=1):
        level = max(0.02, min(1.0, float(level)))  # Keep minimum at 0.02 so screen is not totally pitch black
        with self._lock:
            self.current_brightness = level
            if self.ds:
                try:
                    self._set_b(display_id, ctypes.c_float(level))
                except Exception as e:
                    print(f"[BrightnessController] Error setting brightness: {e}")

    def fade_to(self, target_level, duration=2.0, display_id=1):
        """Fade brightness smoothly to target_level over duration seconds."""
        self._stop_fade.set()
        if self._fade_thread and self._fade_thread.is_alive():
            self._fade_thread.join()
            
        self._stop_fade.clear()
        
        def _fade_worker():
            start_b = self.current_brightness
            end_b = max(0.02, min(1.0, float(target_level)))
            steps = int(max(10, duration * 30))  # 30 fps smooth update
            sleep_time = duration / steps
            
            for i in range(1, steps + 1):
                if self._stop_fade.is_set():
                    break
                t = i / steps
                # Smooth ease-in-out quadratic curve
                ease_t = 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
                current = start_b + (end_b - start_b) * ease_t
                self.set_brightness(current, display_id=display_id)
                time.sleep(sleep_time)

        self._fade_thread = threading.Thread(target=_fade_worker, daemon=True)
        self._fade_thread.start()

    def restore_brightness(self):
        """Restore screen brightness to original level immediately."""
        self._stop_fade.set()
        print(f"[BrightnessController] Restoring original brightness: {self.original_brightness:.2f}")
        self.set_brightness(self.original_brightness)


if __name__ == "__main__":
    ctrl = BrightnessController()
    print(f"Current brightness: {ctrl.original_brightness}")
    print("Testing fade out to 0.1 over 2 seconds...")
    ctrl.fade_to(0.1, duration=2.0)
    time.sleep(3.0)
    print("Testing restore...")
    ctrl.restore_brightness()
    print("Test finished.")
