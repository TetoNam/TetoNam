import cv2
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO
import time
import afb
import serial
import threading
import signal
import sys
import queue
import os
import socket
import glob

# --- 최적화 설정 ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
YOLO_PROCESS_INTERVAL = 8
MAX_STREAM_FPS = 30

# --- 시리얼 포트 초기화 ---
def init_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
        time.sleep(1.5)
        print(f"✓ 지정된 시리얼 포트 ({SERIAL_PORT}) 연결 성공")
        ser.flushInput()
        return ser
    except serial.SerialException:
        print(f"✗ 지정된 포트({SERIAL_PORT}) 연결 실패. 다른 포트를 검색합니다...")
        possible_ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
        for port in possible_ports:
            try:
                ser = serial.Serial(port, BAUD_RATE, timeout=0.05)
                time.sleep(1.5)
                print(f"✓ 자동 검색된 시리얼 포트 ({port}) 연결 성공")
                ser.flushInput()
                return ser
            except Exception:
                continue
    print("✗ 경고: 어떤 시리얼 포트도 연결할 수 없습니다.")
    return None

ser = init_serial()

# --- 모델 및 카메라 초기화 (이전과 동일) ---
print("🤖 YOLO 모델 로딩 중...")
model = YOLO("/home/pi/braille/last.pt")
print("✓ YOLO 모델 로딩 완료")
try:
    afb.camera.init(640, 480, 30)
    print("✓ 카메라 초기화 성공")
except Exception as e:
    print(f"✗ 치명적 오류: 카메라를 켤 수 없습니다. 오류: {e}")
    sys.exit(1)

app = Flask(__name__)

# --- 전역 변수 및 클래스 (이전과 동일) ---
frame_count = 0
current_frame = None
frame_lock = threading.Lock()
yolo_results = {'last_update': 0}
yolo_lock = threading.Lock()

class HighPerformanceSerial:
    def __init__(self, serial_connection):
        self.ser = serial_connection
        self.lock = threading.Lock()
        self.signal_queue = queue.Queue(maxsize=1)
        self.running = True
    def send_signal(self, signal):
        if not self.ser: return
        try:
            while not self.signal_queue.empty():
                self.signal_queue.get_nowait()
            self.signal_queue.put_nowait(signal)
        except queue.Full: pass
    def communication_worker(self):
        while self.running:
            try:
                if not self.ser:
                    time.sleep(0.1)
                    continue
                signal_data = self.signal_queue.get(timeout=0.2)
                with self.lock:
                    message = signal_data + '\n'
                    self.ser.write(message.encode('utf-8'))
                    self.ser.flush()
            except queue.Empty: continue
            except Exception as e:
                print(f"통신 스레드 오류: {e}")
                time.sleep(0.1)

hp_serial = HighPerformanceSerial(ser)

class YOLOProcessor:
    def __init__(self, model):
        self.model = model
        self.running = True
        self.camera_ref_point = (320, 460)
        self.last_sent_signal = None 
    def process_yolo_worker(self):
        global frame_count, yolo_results
        while self.running:
            try:
                if frame_count > 0 and frame_count % YOLO_PROCESS_INTERVAL == 0:
                    with frame_lock:
                        if current_frame is None: continue
                        frame_to_process = current_frame.copy()
                    
                    frame_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                    results = self.model.predict(frame_rgb, imgsz=640, conf=0.3, verbose=False)[0]
                    new_results = self.analyze_results(results)
                    
                    with yolo_lock:
                        yolo_results.update(new_results)
                        yolo_results['last_update'] = time.time()
                    
                    current_signal = new_results.get('current_signal')
                    if current_signal and current_signal != self.last_sent_signal:
                        print(f"✅ 상태 변경 감지: '{self.last_sent_signal}' -> '{current_signal}'. 신호 전송!")
                        hp_serial.send_signal(current_signal)
                        self.last_sent_signal = current_signal
                
                time.sleep(0.01)
            except Exception as e:
                print(f"YOLO 처리 오류: {e}")
                time.sleep(0.1)
    
    def analyze_results(self, results):
        stop_centroid, lowest_go_point = None, None
        all_go_bottom_centers = []
        current_signal = 'S'
        guidance_text = ''
        if results.masks:
            class_names = self.model.names
            go_masks, stop_masks = [], []
            for i, box in enumerate(results.boxes):
                if len(results.masks) > i:
                    cls_id = int(box.cls[0])
                    class_name = class_names[cls_id]
                    if class_name == 'go': go_masks.append(results.masks.xy[i])
                    elif class_name == 'stop': stop_masks.append(results.masks.xy[i])
            
            if stop_masks:
                mask_points = stop_masks[0].astype(int)
                M = cv2.moments(mask_points)
                if M["m00"] != 0:
                    stop_centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            max_y = -1
            for go_contour in go_masks:
                points_sorted = sorted(go_contour, key=lambda p: p[1], reverse=True)
                if len(points_sorted) >= 2:
                    p1, p2 = points_sorted[0], points_sorted[1]
                    target_point = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
                    all_go_bottom_centers.append(target_point)
                    if target_point[1] > max_y:
                        max_y = target_point[1]
                        lowest_go_point = target_point
            
            if stop_centroid and stop_centroid[1] > 240 and len(all_go_bottom_centers) >= 1:
                num_go = len(all_go_bottom_centers)
                if num_go >= 3:
                    current_signal, guidance_text = ("C", "교차로")
                else:
                    x_diff = sum(gp[0] - stop_centroid[0] for gp in all_go_bottom_centers)
                    current_signal, guidance_text = ("B", "좌회전") if x_diff < 0 else ("W", "우회전")
            elif lowest_go_point:
                dead_zone = 25 
                if lowest_go_point[0] < self.camera_ref_point[0] - dead_zone:
                    current_signal = 'L'
                elif lowest_go_point[0] > self.camera_ref_point[0] + dead_zone:
                    current_signal = 'R'
                else:
                    current_signal = 'S'
        return {'current_signal': current_signal, 'guidance_text': guidance_text}

yolo_processor = YOLOProcessor(model)

# 💡 [추가] 아두이노로부터 메시지를 수신하는 워커 함수
def serial_reader_worker(ser_connection):
    """아두이노로부터 들어오는 메시지를 계속 읽어서 터미널에 출력합니다."""
    while True:
        try:
            if ser_connection and ser_connection.in_waiting > 0:
                # 메시지를 줄 단위로 읽고, utf-8로 디코딩하며, 양쪽 공백을 제거합니다.
                message = ser_connection.readline().decode('utf-8').strip()
                if message: # 빈 메시지가 아닐 경우에만 출력
                    if message == "DFP_OK":
                        print("✅ [아두이노 응답]: DFPlayer 초기화 성공!")
                    elif message == "DFP_FAIL":
                        print("❌ [아두이노 응답]: DFPlayer 초기화 실패! 배선과 SD카드를 확인하세요.")
                    else:
                        print(f"💬 [아두이노 메시지]: {message}")
        except Exception as e:
            print(f"시리얼 읽기 스레드 오류: {e}")
            break # 오류 발생 시 스레드 종료
        time.sleep(0.1)


# --- 웹 서버 및 기타 함수 (이전과 동일) ---
def draw_annotations(frame):
    with yolo_lock: results = yolo_results.copy()
    signal_text = f"Signal: {yolo_processor.last_sent_signal or 'N/A'}"
    cv2.putText(frame, signal_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    if time.time() - results.get('last_update', 0) > 1.5: return frame
    if results.get('stop_centroid'): cv2.circle(frame, results['stop_centroid'], 7, (0, 255, 255), -1)
    for go_point in results.get('all_go_bottom_centers', []): cv2.drawMarker(frame, go_point, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 15, 2)
    if results.get('lowest_go_point'): cv2.line(frame, yolo_processor.camera_ref_point, results['lowest_go_point'], (255, 255, 0), 2)
    if results.get('guidance_text'): cv2.putText(frame, results['guidance_text'], (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 2)
    return frame
def camera_capture_worker():
    global frame_count, current_frame
    while yolo_processor.running:
        try:
            frame = afb.camera.get_image()
            if frame.shape[2] == 4: frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            with frame_lock:
                current_frame = frame
                frame_count += 1
            time.sleep(1/MAX_STREAM_FPS)
        except Exception as e:
            print(f"카메라 캡처 중 오류 발생: {e}")
            time.sleep(0.5)
def generate():
    while True:
        try:
            with frame_lock:
                if current_frame is None: 
                    time.sleep(0.01)
                    continue
                frame_to_stream = current_frame.copy()
            annotated_frame = draw_annotations(frame_to_stream)
            _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1/MAX_STREAM_FPS)
        except Exception as e:
            print(f"스트리밍 오류: {e}")
            break
@app.route('/video')
def video(): return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index(): return '<h1>🚀 Ultra-Fast Haptic Guidance</h1><img src="/video" width="640" height="480">'
def start_all_threads():
    print("🚀 고성능 스레드 시작 중...")
    camera_thread = threading.Thread(target=camera_capture_worker, daemon=True)
    camera_thread.start()
    print("✓ 카메라 스레드 시작")
    yolo_thread = threading.Thread(target=yolo_processor.process_yolo_worker, daemon=True)
    yolo_thread.start()
    print("✓ YOLO 처리 스레드 시작")
    if ser:
        serial_thread = threading.Thread(target=hp_serial.communication_worker, daemon=True)
        serial_thread.start()
        print("✓ 시리얼 통신 스레드 시작")
def cleanup_and_exit(signum, frame):
    print(f"\n🛑 시스템 종료 중...")
    yolo_processor.running = False
    hp_serial.running = False
    if ser:
        try:
            ser.write(b'S\n')
            ser.close()
            print("✓ 시리얼 연결 종료")
        except: pass
    print("👋 시스템 종료 완료")
    sys.exit(0)
def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception: return "127.0.0.1"
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

if __name__ == '__main__':
    try:
        print("🚀 Ultra-Fast 햅틱 가이드 시스템 시작 (상태 관리 적용)!")
        start_all_threads()
        
        # 💡 [추가] 아두이노 메시지 수신 스레드 시작
        if ser:
            reader_thread = threading.Thread(target=serial_reader_worker, args=(ser,), daemon=True)
            reader_thread.start()
            print("✓ Arduino 메시지 수신 스레드 시작됨.")

        time.sleep(2)
        
        ip_address = get_ip_address()
        print("="*50)
        print("🌐 웹 서버 접속 주소 (디버깅용):")
        print(f"  http://{ip_address}:5000")
        print("="*50)
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

    except Exception as e:
        print(f"❌ 시스템 시작 오류: {e}")
        cleanup_and_exit(0, None)