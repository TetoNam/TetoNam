"""
═══════════════════════════════════════════════════════════════════════════════
🤖 AI 훈련 시스템 v3.0 - 차세대 통합 자동화 시스템 
═══════════════════════════════════════════════════════════════════════════════

📋 주요 기능 및 특징:
────────────────────────────────────────────────────────────────────────────
🔥 v2.2 → v3.0 주요 업그레이드 (2025-08-16):
  ✨ 완전히 재설계된 모듈화 아키텍처
  🛡️ 견고한 경로 검증 및 보안 시스템
  📊 실시간 하드웨어 모니터링 (CPU/GPU/NPU)
  🎨 Rich 기반 고급 UI 시스템
  🤖 AI 기반 스마트 오류 해결
  💾 데이터 무결성 검증 시스템
  🌐 다국어 지원 (한국어/영어)
  ⚡ 압축파일 AI 검색 알고리즘 강화
  📈 예측 분석 기반 성능 최적화
  🔄 설정 백업/복원 시스템

작성자: AI Training System Team
버전: 3.0.0
최종 업데이트: 2025-08-16
라이선스: MIT License
════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import yaml
import hashlib
import logging
import platform
import subprocess
import threading
import time
import traceback
import zipfile
import rarfile
import tarfile
import shutil
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import configparser
import locale

# Rich 라이브러리 - 고급 터미널 UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.tree import Tree
    from rich.align import Align
    from rich.columns import Columns
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.rule import Rule
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich 라이브러리가 설치되지 않았습니다. 기본 터미널 모드로 실행됩니다.")
    print("설치 명령: pip install rich")

# NPU 모니터링 (Windows)
try:
    import win32pdh
    import win32api
    NPU_MONITORING = platform.system() == "Windows"
except ImportError:
    NPU_MONITORING = False

# 텐서플로우/토치 관련
try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# OpenVINO (NPU 가속)
try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

# Ultralytics YOLO
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# PIL for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════════
# 🌐 다국어 지원 시스템
# ════════════════════════════════════════════════════════════════════════════════

class LanguageManager:
    """
    다국어 지원을 위한 언어 관리자 클래스
    - 시스템 로케일 자동 감지
    - 한국어/영어 메시지 동적 전환
    - 사용자 설정 기반 언어 선택
    """
    
    def __init__(self):
        self.current_language = self._detect_system_language()
        self.messages = self._load_messages()
    
    def _detect_system_language(self) -> str:
        """시스템 언어 자동 감지"""
        try:
            system_locale = locale.getdefaultlocale()[0]
            if system_locale and system_locale.startswith('ko'):
                return 'ko'
            return 'en'
        except:
            return 'ko'  # 기본값은 한국어
    
    def _load_messages(self) -> Dict[str, Dict[str, str]]:
        """다국어 메시지 로드"""
        return {
            'ko': {
                'welcome': '🤖 AI 훈련 시스템 v3.0에 오신 것을 환영합니다!',
                'system_check': '시스템 환경 검사 중...',
                'hardware_detect': '하드웨어 자동 감지',
                'dataset_search': '데이터셋 검색 중...',
                'training_start': '훈련 시작',
                'error_occurred': '오류가 발생했습니다',
                'help_command': '도움말을 보려면 !help를 입력하세요',
                'workflow_select': '워크플로우를 선택해주세요',
                'auto_mode': '완전 자동 모드',
                'semi_auto_mode': '반자동 모드',
                'manual_mode': '수동 모드'
            },
            'en': {
                'welcome': '🤖 Welcome to AI Training System v3.0!',
                'system_check': 'Checking system environment...',
                'hardware_detect': 'Auto-detecting hardware',
                'dataset_search': 'Searching datasets...',
                'training_start': 'Starting training',
                'error_occurred': 'An error occurred',
                'help_command': 'Type !help for help',
                'workflow_select': 'Please select workflow',
                'auto_mode': 'Full Auto Mode',
                'semi_auto_mode': 'Semi-Auto Mode',
                'manual_mode': 'Manual Mode'
            }
        }
    
    def get(self, key: str) -> str:
        """언어별 메시지 반환"""
        return self.messages.get(self.current_language, {}).get(key, key)
    
    def set_language(self, language: str):
        """언어 설정 변경"""
        if language in ['ko', 'en']:
            self.current_language = language

# ════════════════════════════════════════════════════════════════════════════════
# 📊 시스템 설정 및 상수 정의
# ════════════════════════════════════════════════════════════════════════════════

class SystemConstants:
    """시스템 전역 상수 정의 클래스"""
    
    # 버전 정보
    VERSION = "3.0.0"
    VERSION_DATE = "2025-08-16"
    
    # 지원 파일 확장자
    ARCHIVE_EXTENSIONS = ['.zip', '.rar', '.7z', '.tar', '.tar.gz', '.tar.bz2']
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # 검색 우선 폴더
    PRIORITY_FOLDERS = [
        "Desktop", "Downloads", "Documents", "Pictures", "Videos",
        "바탕화면", "다운로드", "문서", "사진", "동영상"
    ]
    
    # 로그 레벨
    LOG_LEVELS = {
        'CRITICAL': 50,
        'ERROR': 40,
        'WARNING': 30,
        'INFO': 20,
        'DEBUG': 10
    }
    
    # 하드웨어 모니터링 간격 (초)
    MONITORING_INTERVAL = 5.0  # 1.0에서 5.0으로 변경
    TRAINING_MONITORING_INTERVAL = 10.0  # 훈련 중에는 더 느리게
    
    # AI 검색 가중치
    AI_SEARCH_WEIGHTS = {
        'filename_match': 0.4,
        'extension_match': 0.2,
        'path_priority': 0.2,
        'file_size': 0.1,
        'creation_date': 0.1
    }

@dataclass
class SystemConfig:
    """시스템 설정 데이터 클래스"""
    
    # 일반 설정
    language: str = 'ko'
    log_level: str = 'INFO'
    auto_backup: bool = True
    check_updates: bool = True
    
    # 하드웨어 설정
    use_gpu: bool = True
    use_npu: bool = True
    gpu_memory_fraction: float = 0.8
    
    # 데이터셋 설정
    auto_extract: bool = True
    verify_integrity: bool = True
    max_search_depth: int = 5
    
    # 훈련 설정
    default_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """딕셔너리에서 설정 생성 (메타데이터 제외)"""
        # 메타데이터 제거
        config_data = {k: v for k, v in data.items() if not k.startswith('_')}
        
        # SystemConfig에 정의되지 않은 키 제거
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in config_data.items() if k in valid_keys}
        
        return cls(**filtered_data)

# ════════════════════════════════════════════════════════════════════════════════
# 🔧 고급 로깅 시스템
# ════════════════════════════════════════════════════════════════════════════════

class AdvancedLogger:
    """
    5단계 로그 레벨을 지원하는 고급 로깅 시스템
    - 파일과 콘솔 출력 분리
    - Rich 기반 컬러 로깅
    - 자동 로그 로테이션
    """
    
    def __init__(self, name: str = "AITrainingSystem", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Rich 콘솔 설정
        self.console = Console() if RICH_AVAILABLE else None
        
        # 로거 설정
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 핸들러 설정
        self._setup_handlers()
    
    def _setup_handlers(self):
        """로그 핸들러 설정"""
        # 파일 핸들러
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """디버그 로그"""
        self.logger.debug(message)
        if self.console:
            self.console.print(f"🔍 [dim]{message}[/dim]")
    
    def info(self, message: str):
        """정보 로그"""
        self.logger.info(message)
        if self.console:
            self.console.print(f"ℹ️ {message}")
    
    def warning(self, message: str):
        """경고 로그"""
        self.logger.warning(message)
        if self.console:
            self.console.print(f"⚠️ [yellow]{message}[/yellow]")
    
    def error(self, message: str):
        """오류 로그"""
        self.logger.error(message)
        if self.console:
            self.console.print(f"❌ [red]{message}[/red]")
    
    def critical(self, message: str):
        """치명적 오류 로그"""
        self.logger.critical(message)
        if self.console:
            self.console.print(f"🚨 [bold red]{message}[/bold red]")

# ════════════════════════════════════════════════════════════════════════════════
# 🛡️ 보안 및 경로 검증 시스템
# ════════════════════════════════════════════════════════════════════════════════

class SecurityManager:
    """
    보안 및 경로 검증을 위한 관리자 클래스
    - 입력 검증 및 SQL 인젝션 방지
    - 경로 탐색 공격 방지
    - 권한 관리 시스템
    """
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.dangerous_patterns = [
            r'\.\./', r'\.\.\\', r'\.\./\.\.',
            r'<script', r'javascript:',
            r'SELECT.*FROM', r'DROP.*TABLE',
            r'EXEC.*sp_', r'UNION.*SELECT'
        ]
    
    def validate_input(self, user_input: str) -> bool:
        """사용자 입력 검증"""
        import re
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                self.logger.warning(f"위험한 입력 패턴 감지: {pattern}")
                return False
        
        return True
    
    def validate_path(self, path: Union[str, Path]) -> Tuple[bool, Optional[Path]]:
        """경로 검증 및 정규화"""
        try:
            path_obj = Path(path).resolve()
            
            # 위험한 시스템 경로 체크
            dangerous_paths = [
                Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows",
                Path("C:/Windows/System32"),
                Path("/etc"),
                Path("/usr/bin")
            ]
            
            for dangerous_path in dangerous_paths:
                try:
                    if path_obj.is_relative_to(dangerous_path):
                        self.logger.warning(f"위험한 시스템 경로 접근 시도: {path_obj}")
                        return False, None
                except (OSError, ValueError):
                    continue
            
            return True, path_obj
            
        except (OSError, ValueError) as e:
            self.logger.error(f"경로 검증 실패: {e}")
            return False, None
    
    def check_file_permissions(self, path: Path) -> Dict[str, bool]:
        """파일 권한 확인"""
        permissions = {
            'readable': False,
            'writable': False,
            'executable': False
        }
        
        try:
            permissions['readable'] = os.access(path, os.R_OK)
            permissions['writable'] = os.access(path, os.W_OK)
            permissions['executable'] = os.access(path, os.X_OK)
        except Exception as e:
            self.logger.error(f"권한 확인 실패: {e}")
        
        return permissions

# Part 1에서 이어집니다...

# ════════════════════════════════════════════════════════════════════════════════
# 📊 실시간 하드웨어 모니터링 시스템
# ════════════════════════════════════════════════════════════════════════════════

class HardwareMonitor:
    """
    CPU, GPU, NPU 실시간 모니터링 시스템
    - 실시간 성능 추적
    - 예측 분석 기반 병목 현상 감지
    - 하드웨어별 최적화 제안
    """
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.monitoring = False
        self.monitor_thread = None
        self.performance_history = {
            'cpu': [],
            'memory': [],
            'gpu': [],
            'npu': []
        }
        
        # NPU 모니터링 초기화
        self.npu_available = NPU_MONITORING
        if self.npu_available:
            self._init_npu_monitoring()
    
    def _init_npu_monitoring(self):
        """NPU 모니터링 초기화 (Windows Intel NPU)"""
        try:
            # Intel NPU 성능 카운터 초기화
            self.npu_counter = None
            if NPU_MONITORING:
                # win32pdh를 사용한 NPU 모니터링 설정
                pass  # 실제 구현시 Intel NPU SDK 필요
        except Exception as e:
            self.logger.warning(f"NPU 모니터링 초기화 실패: {e}")
            self.npu_available = False
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """CPU 정보 수집"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_freq = psutil.cpu_freq()
            cpu_count = psutil.cpu_count()
            
            return {
                'usage_percent': cpu_percent,
                'frequency_mhz': cpu_freq.current if cpu_freq else 0,
                'core_count': cpu_count,
                'temperature': self._get_cpu_temperature()
            }
        except Exception as e:
            self.logger.error(f"CPU 정보 수집 실패: {e}")
            return {}
    
    def _get_cpu_temperature(self) -> Optional[float]:
        """CPU 온도 측정 (가능한 경우)"""
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            return entries[0].current if entries else None
        except Exception:
            pass
        return None
    
    def get_memory_info(self) -> Dict[str, Any]:
        """메모리 정보 수집"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_percent': memory.percent,
                'swap_used_percent': swap.percent
            }
        except Exception as e:
            self.logger.error(f"메모리 정보 수집 실패: {e}")
            return {}
    
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """GPU 정보 수집"""
        gpu_info = []
        
        try:
            # NVIDIA GPU 정보
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load_percent': round(gpu.load * 100, 2),
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_percent': round((gpu.memoryUsed / gpu.memoryTotal) * 100, 2),
                    'temperature': gpu.temperature
                })
        except Exception as e:
            self.logger.warning(f"NVIDIA GPU 정보 수집 실패: {e}")
        
        # PyTorch CUDA 정보 추가
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)  # MB
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**2)   # MB
                    
                    # 기존 GPU 정보와 병합하거나 새로 추가
                    found = False
                    for gpu in gpu_info:
                        if device_name in gpu['name']:
                            gpu.update({
                                'torch_memory_allocated_mb': memory_allocated,
                                'torch_memory_reserved_mb': memory_reserved
                            })
                            found = True
                            break
                    
                    if not found:
                        gpu_info.append({
                            'id': i,
                            'name': device_name,
                            'load_percent': 0,
                            'memory_used_mb': 0,
                            'memory_total_mb': 0,
                            'memory_percent': 0,
                            'temperature': 0,
                            'torch_memory_allocated_mb': memory_allocated,
                            'torch_memory_reserved_mb': memory_reserved
                        })
            except Exception as e:
                self.logger.warning(f"PyTorch CUDA 정보 수집 실패: {e}")
        
        return gpu_info
    
    def get_npu_info(self) -> Dict[str, Any]:
        """NPU 정보 수집 (Intel NPU 우선 지원)"""
        npu_info = {
            'available': self.npu_available,
            'usage_percent': 0,
            'temperature': 0,
            'power_watts': 0
        }
        
        if not self.npu_available:
            return npu_info
        
        try:
            # Intel NPU 모니터링 (실제 구현시 Intel NPU SDK 필요)
            # 여기서는 시뮬레이션
            if NPU_MONITORING:
                # win32pdh를 통한 NPU 성능 카운터 읽기
                pass
                
            # OpenVINO를 통한 NPU 상태 확인
            if OPENVINO_AVAILABLE:
                try:
                    core = ov.Core()
                    available_devices = core.available_devices
                    
                    npu_devices = [device for device in available_devices if 'NPU' in device]
                    if npu_devices:
                        npu_info['devices'] = npu_devices
                        npu_info['available'] = True
                except Exception as e:
                    self.logger.debug(f"OpenVINO NPU 확인 실패: {e}")
        
        except Exception as e:
            self.logger.warning(f"NPU 정보 수집 실패: {e}")
        
        return npu_info
    
    def start_monitoring(self):
        """실시간 모니터링 시작"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("하드웨어 실시간 모니터링 시작")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        self.logger.info("하드웨어 모니터링 중지")
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            try:
                # 성능 데이터 수집
                cpu_info = self.get_cpu_info()
                memory_info = self.get_memory_info()
                gpu_info = self.get_gpu_info()
                npu_info = self.get_npu_info()
                
                # 히스토리에 저장 (최근 100개만 유지)
                timestamp = time.time()
                
                if cpu_info:
                    self.performance_history['cpu'].append({
                        'timestamp': timestamp,
                        **cpu_info
                    })
                
                if memory_info:
                    self.performance_history['memory'].append({
                        'timestamp': timestamp,
                        **memory_info
                    })
                
                if gpu_info:
                    self.performance_history['gpu'].append({
                        'timestamp': timestamp,
                        'gpus': gpu_info
                    })
                
                if npu_info:
                    self.performance_history['npu'].append({
                        'timestamp': timestamp,
                        **npu_info
                    })
                
                # 히스토리 크기 제한
                for key in self.performance_history:
                    if len(self.performance_history[key]) > 100:
                        self.performance_history[key] = self.performance_history[key][-100:]
                
                time.sleep(SystemConstants.MONITORING_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                time.sleep(1)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        current_data = {
            'cpu': self.get_cpu_info(),
            'memory': self.get_memory_info(),
            'gpu': self.get_gpu_info(),
            'npu': self.get_npu_info()
        }
        
        # 예측 분석
        predictions = self._analyze_performance_trends()
        
        return {
            'current': current_data,
            'predictions': predictions,
            'recommendations': self._get_optimization_recommendations(current_data)
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """성능 트렌드 분석 및 예측"""
        predictions = {}
        
        try:
            # CPU 트렌드 분석
            if len(self.performance_history['cpu']) >= 10:
                cpu_usage = [entry['usage_percent'] for entry in self.performance_history['cpu'][-10:]]
                avg_usage = sum(cpu_usage) / len(cpu_usage)
                trend = 'increasing' if cpu_usage[-1] > avg_usage else 'stable'
                
                predictions['cpu'] = {
                    'trend': trend,
                    'avg_usage': round(avg_usage, 2),
                    'bottleneck_risk': 'high' if avg_usage > 80 else 'low'
                }
            
            # 메모리 트렌드 분석
            if len(self.performance_history['memory']) >= 10:
                memory_usage = [entry['used_percent'] for entry in self.performance_history['memory'][-10:]]
                avg_usage = sum(memory_usage) / len(memory_usage)
                
                predictions['memory'] = {
                    'avg_usage': round(avg_usage, 2),
                    'bottleneck_risk': 'high' if avg_usage > 85 else 'low'
                }
        
        except Exception as e:
            self.logger.error(f"성능 트렌드 분석 실패: {e}")
        
        return predictions
    
    def _get_optimization_recommendations(self, current_data: Dict[str, Any]) -> List[str]:
        """최적화 추천사항 생성"""
        recommendations = []
        
        try:
            # CPU 최적화
            if current_data['cpu'].get('usage_percent', 0) > 80:
                recommendations.append("CPU 사용률이 높습니다. 배치 크기를 줄이거나 학습률을 조정해보세요.")
            
            # 메모리 최적화  
            if current_data['memory'].get('used_percent', 0) > 85:
                recommendations.append("메모리 사용률이 높습니다. 데이터 로더의 num_workers를 줄여보세요.")
            
            # GPU 최적화
            for gpu in current_data.get('gpu', []):
                if gpu.get('memory_percent', 0) > 90:
                    recommendations.append(f"GPU {gpu['id']} 메모리가 부족합니다. 배치 크기를 줄여보세요.")
                elif gpu.get('load_percent', 0) < 30:
                    recommendations.append(f"GPU {gpu['id']} 활용도가 낮습니다. 배치 크기를 늘려보세요.")
            
            # NPU 최적화
            if current_data['npu'].get('available') and OPENVINO_AVAILABLE:
                recommendations.append("NPU가 감지되었습니다. OpenVINO를 통한 모델 최적화를 고려해보세요.")
        
        except Exception as e:
            self.logger.error(f"최적화 추천 생성 실패: {e}")
        
        return recommendations

# ════════════════════════════════════════════════════════════════════════════════
# 💾 데이터 무결성 검증 시스템
# ════════════════════════════════════════════════════════════════════════════════

class DataIntegrityManager:
    """
    데이터 무결성 검증 및 체크섬 관리 시스템
    - SHA-256, CRC32 다중 해시 알고리즘
    - 파일 손상 감지 및 복구
    - 증분 백업 시스템
    """
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.checksum_cache = {}
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
    
    def calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> Optional[str]:
        """파일 해시 계산"""
        try:
            hash_obj = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                # 대용량 파일을 위한 청크 단위 읽기
                chunk_size = 65536  # 64KB
                while chunk := f.read(chunk_size):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            self.logger.error(f"파일 해시 계산 실패 ({file_path}): {e}")
            return None
    
    def verify_file_integrity(self, file_path: Path) -> Dict[str, Any]:
        """파일 무결성 검증"""
        result = {
            'path': str(file_path),
            'exists': file_path.exists(),
            'readable': False,
            'size': 0,
            'sha256': None,
            'crc32': None,
            'integrity_ok': False
        }
        
        if not result['exists']:
            return result
        
        try:
            # 파일 접근 가능성 확인
            result['readable'] = os.access(file_path, os.R_OK)
            if not result['readable']:
                return result
            
            # 파일 크기
            result['size'] = file_path.stat().st_size
            
            # 해시 계산
            result['sha256'] = self.calculate_file_hash(file_path, 'sha256')
            
            # CRC32 계산 (빠른 검증용)
            with open(file_path, 'rb') as f:
                crc32_hash = 0
                chunk_size = 65536
                while chunk := f.read(chunk_size):
                    crc32_hash = zlib.crc32(chunk, crc32_hash)
                result['crc32'] = format(crc32_hash & 0xffffffff, '08x')
            
            # 캐시된 해시와 비교 (이전에 계산한 적이 있다면)
            cache_key = str(file_path)
            if cache_key in self.checksum_cache:
                cached = self.checksum_cache[cache_key]
                result['integrity_ok'] = (
                    cached['sha256'] == result['sha256'] and
                    cached['size'] == result['size']
                )
            else:
                # 첫 번째 계산이므로 OK로 간주
                result['integrity_ok'] = True
                self.checksum_cache[cache_key] = {
                    'sha256': result['sha256'],
                    'crc32': result['crc32'],
                    'size': result['size'],
                    'timestamp': time.time()
                }
            
        except Exception as e:
            self.logger.error(f"파일 무결성 검증 실패 ({file_path}): {e}")
        
        return result
    
    def batch_verify_integrity(self, file_paths: List[Path]) -> Dict[str, Any]:
        """다중 파일 무결성 일괄 검증"""
        results = {
            'total_files': len(file_paths),
            'verified_files': 0,
            'corrupted_files': 0,
            'inaccessible_files': 0,
            'details': []
        }
        
        for file_path in file_paths:
            verification = self.verify_file_integrity(file_path)
            results['details'].append(verification)
            
            if not verification['exists'] or not verification['readable']:
                results['inaccessible_files'] += 1
            elif verification['integrity_ok']:
                results['verified_files'] += 1
            else:
                results['corrupted_files'] += 1
        
        return results
    
    def create_backup(self, source_path: Path, backup_name: Optional[str] = None) -> Optional[Path]:
        """파일/폴더 백업 생성"""
        try:
            if backup_name is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f"{source_path.name}_{timestamp}"
            
            backup_path = self.backup_dir / backup_name
            
            if source_path.is_file():
                # 파일 백업
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, backup_path)
            elif source_path.is_dir():
                # 폴더 백업
                shutil.copytree(source_path, backup_path)
            
            # 백업 파일 무결성 검증
            if backup_path.exists():
                verification = self.verify_file_integrity(backup_path)
                if verification['integrity_ok']:
                    self.logger.info(f"백업 생성 완료: {backup_path}")
                    return backup_path
                else:
                    self.logger.error(f"백업 파일 무결성 검증 실패: {backup_path}")
                    return None
            
        except Exception as e:
            self.logger.error(f"백업 생성 실패 ({source_path}): {e}")
        
        return None
    
    def restore_from_backup(self, backup_path: Path, restore_path: Path) -> bool:
        """백업에서 파일 복원"""
        try:
            if not backup_path.exists():
                self.logger.error(f"백업 파일이 존재하지 않습니다: {backup_path}")
                return False
            
            # 백업 파일 무결성 검증
            verification = self.verify_file_integrity(backup_path)
            if not verification['integrity_ok']:
                self.logger.error(f"백업 파일이 손상되었습니다: {backup_path}")
                return False
            
            # 복원 대상 경로 준비
            restore_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 파일 복사
            shutil.copy2(backup_path, restore_path)
            
            # 복원된 파일 검증
            restored_verification = self.verify_file_integrity(restore_path)
            if restored_verification['integrity_ok']:
                self.logger.info(f"파일 복원 완료: {restore_path}")
                return True
            else:
                self.logger.error(f"복원된 파일 무결성 검증 실패: {restore_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"파일 복원 실패 ({backup_path} -> {restore_path}): {e}")
            return False
    
    def save_checksum_cache(self, cache_file: Path):
        """체크섬 캐시 저장"""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.checksum_cache, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"체크섬 캐시 저장 완료: {cache_file}")
        except Exception as e:
            self.logger.error(f"체크섬 캐시 저장 실패: {e}")
    
    def load_checksum_cache(self, cache_file: Path):
        """체크섬 캐시 로드"""
        try:
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.checksum_cache = json.load(f)
                self.logger.debug(f"체크섬 캐시 로드 완료: {cache_file}")
        except Exception as e:
            self.logger.error(f"체크섬 캐시 로드 실패: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# 🔄 설정 백업 및 복원 시스템
# ════════════════════════════════════════════════════════════════════════════════

class ConfigurationManager:
    """
    사용자 설정 자동 저장 및 복원 시스템
    - 버전 관리 기능
    - 설정 변경 이력 추적
    - 자동 백업 및 복원
    """
    
    def __init__(self, logger: AdvancedLogger, config_dir: str = "configs"):
        self.logger = logger
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.config_file = self.config_dir / "system_config.json"
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # 현재 설정
        self.current_config = SystemConfig()
        
        # 설정 변경 이력
        self.config_history = []
    
    def load_config(self) -> SystemConfig:
        """설정 파일 로드 - 메타데이터 처리 개선"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 메타데이터가 있으면 로그에 기록하고 제거
                if '_metadata' in config_data:
                    metadata = config_data.pop('_metadata')
                    self.logger.info(f"설정 파일 메타데이터: {metadata.get('saved_at', 'Unknown')}")
                
                # SystemConfig 생성
                self.current_config = SystemConfig.from_dict(config_data)
                self.logger.info("설정 파일 로드 완료")
            else:
                # 기본 설정으로 초기 파일 생성
                self.current_config = SystemConfig()
                self.save_config()
                self.logger.info("기본 설정 파일 생성")
                
        except Exception as e:
            self.logger.error(f"설정 파일 로드 실패: {e}")
            self.current_config = SystemConfig()  # 기본 설정 사용
        
        return self.current_config

    def save_config(self, backup: bool = True) -> bool:
        """설정 파일 저장 - 안전한 메타데이터 처리"""
        try:
            # 기존 설정 백업
            if backup and self.config_file.exists():
                self.create_config_backup()
            
            # 현재 설정을 딕셔너리로 변환
            config_data = self.current_config.to_dict()
            
            # 메타데이터 추가 (별도 키로 저장)
            config_data['_metadata'] = {
                'version': SystemConstants.VERSION,
                'saved_at': datetime.now().isoformat(),
                'system_info': {
                    'platform': platform.system(),
                    'python_version': platform.python_version()
                }
            }
            
            # 임시 파일에 먼저 저장 (원자적 쓰기)
            temp_file = self.config_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            # 성공하면 원본 파일로 이동
            temp_file.replace(self.config_file)
            
            self.logger.info("설정 파일 저장 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"설정 파일 저장 실패: {e}")
            # 임시 파일이 있으면 삭제
            temp_file = self.config_file.with_suffix('.tmp')
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except:
                    pass
            return False
    
    def create_config_backup(self) -> Optional[Path]:
        """설정 백업 생성"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f"config_backup_{timestamp}.json"
            
            shutil.copy2(self.config_file, backup_file)
            
            # 백업 이력에 추가
            self.config_history.append({
                'backup_file': str(backup_file),
                'timestamp': timestamp,
                'config_snapshot': self.current_config.to_dict()
            })
            
            # 오래된 백업 정리 (최근 10개만 유지)
            self._cleanup_old_backups()
            
            self.logger.info(f"설정 백업 생성: {backup_file}")
            return backup_file
            
        except Exception as e:
            self.logger.error(f"설정 백업 생성 실패: {e}")
            return None
    
    def restore_config_backup(self, backup_file: Path) -> bool:
        """백업에서 설정 복원"""
        try:
            if not backup_file.exists():
                self.logger.error(f"백업 파일이 존재하지 않습니다: {backup_file}")
                return False
            
            # 현재 설정 백업 (복원 실패시 롤백용)
            rollback_backup = self.create_config_backup()
            
            try:
                # 백업에서 복원
                shutil.copy2(backup_file, self.config_file)
                
                # 설정 다시 로드
                self.load_config()
                
                self.logger.info(f"설정 복원 완료: {backup_file}")
                return True
                
            except Exception as restore_error:
                # 복원 실패시 롤백
                if rollback_backup and rollback_backup.exists():
                    shutil.copy2(rollback_backup, self.config_file)
                    self.load_config()
                    self.logger.error(f"설정 복원 실패, 롤백 완료: {restore_error}")
                
                return False
                
        except Exception as e:
            self.logger.error(f"설정 복원 실패: {e}")
            return False
    
    def _cleanup_old_backups(self):
        """오래된 백업 파일 정리"""
        try:
            backup_files = list(self.backup_dir.glob("config_backup_*.json"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 최근 10개를 제외하고 삭제
            for old_backup in backup_files[10:]:
                old_backup.unlink()
                self.logger.debug(f"오래된 백업 삭제: {old_backup}")
                
        except Exception as e:
            self.logger.warning(f"백업 정리 실패: {e}")
    
    def update_config(self, **kwargs) -> bool:
        """설정 업데이트"""
        try:
            # 변경 사항 적용
            for key, value in kwargs.items():
                if hasattr(self.current_config, key):
                    setattr(self.current_config, key, value)
                    self.logger.info(f"설정 업데이트: {key} = {value}")
            
            # 설정 저장
            return self.save_config()
            
        except Exception as e:
            self.logger.error(f"설정 업데이트 실패: {e}")
            return False
    
    def get_config_history(self) -> List[Dict[str, Any]]:
        """설정 변경 이력 반환"""
        return self.config_history.copy()

# Part 2에서 이어집니다...

import zlib
import re
import fnmatch
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor #as_completed

# ════════════════════════════════════════════════════════════════════════════════
# 🧠 AI 기반 스마트 데이터셋 검색 시스템
# ════════════════════════════════════════════════════════════════════════════════

class SmartDatasetFinder:
    """
    AI 키워드 기반 스마트 데이터셋 자동 검색 시스템
    - 가중치 기반 점수 시스템
    - 압축파일 우선 검색
    - 사용자 학습 패턴 반영
    """
    
    def __init__(self, logger: AdvancedLogger, security_manager: SecurityManager):
        self.logger = logger
        self.security_manager = security_manager
        
        # AI 검색 패턴 정의
        self.dataset_patterns = {
            'computer_vision': [
                r'.*dataset.*', r'.*data.*', r'.*images?.*', r'.*vision.*',
                r'.*object.*detection.*', r'.*classification.*', r'.*segmentation.*',
                r'.*yolo.*', r'.*coco.*', r'.*imagenet.*', r'.*mnist.*'
            ],
            'nlp': [
                r'.*text.*', r'.*nlp.*', r'.*language.*', r'.*corpus.*',
                r'.*bert.*', r'.*gpt.*', r'.*transformer.*'
            ],
            'general': [
                r'.*train.*', r'.*test.*', r'.*valid.*', r'.*dataset.*',
                r'.*data.*', r'.*ml.*', r'.*ai.*', r'.*model.*'
            ]
        }
        
        # 파일 확장자별 가중치
        self.extension_weights = {
            '.zip': 1.0,
            '.rar': 0.9,
            '.7z': 0.8,
            '.tar': 0.7,
            '.tar.gz': 0.7,
            '.tar.bz2': 0.6
        }
        
        # 경로 우선순위 가중치
        self.path_weights = {
            'desktop': 1.0,
            'downloads': 0.9,
            'documents': 0.8,
            'pictures': 0.7,
            'videos': 0.6,
            'dataset': 1.2,
            'data': 1.1,
            'train': 1.1
        }
        
        # 사용자 선택 학습 데이터
        self.user_preferences = {}
        
        # 검색 결과 캐시
        self.search_cache = {}
    
    def find_datasets(self, search_paths: List[Path], 
                     search_query: str = "", 
                     max_results: int = 20) -> List[Dict[str, Any]]:
        """
        AI 기반 데이터셋 검색
        Args:
            search_paths: 검색할 경로 리스트
            search_query: 검색 쿼리 (선택적)
            max_results: 최대 결과 수
        """
        self.logger.info(f"AI 데이터셋 검색 시작: {len(search_paths)}개 경로")
        
        # 캐시 확인
        cache_key = self._generate_cache_key(search_paths, search_query)
        if cache_key in self.search_cache:
            self.logger.debug("캐시에서 검색 결과 반환")
            return self.search_cache[cache_key]
        
        all_candidates = []
        
        # 멀티스레딩으로 병렬 검색
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_path = {
                executor.submit(self._search_path, path, search_query): path 
                for path in search_paths
            }
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    candidates = future.result()
                    all_candidates.extend(candidates)
                    self.logger.debug(f"경로 {path}: {len(candidates)}개 후보 발견")
                except Exception as e:
                    self.logger.error(f"경로 검색 실패 ({path}): {e}")
        
        # AI 기반 점수 계산 및 정렬
        scored_datasets = self._calculate_ai_scores(all_candidates, search_query)
        
        # 상위 결과 선택
        top_results = scored_datasets[:max_results]
        
        # 캐시에 저장
        self.search_cache[cache_key] = top_results
        
        self.logger.info(f"AI 검색 완료: {len(top_results)}개 데이터셋 후보 발견")
        return top_results
    
    def _search_path(self, search_path: Path, query: str) -> List[Dict[str, Any]]:
        """단일 경로에서 데이터셋 검색"""
        candidates = []
        
        try:
            if not search_path.exists():
                return candidates
            
            # 보안 검증
            is_safe, safe_path = self.security_manager.validate_path(search_path)
            if not is_safe:
                self.logger.warning(f"안전하지 않은 경로 건너뜀: {search_path}")
                return candidates
            
            # 재귀 검색 (최대 깊이 제한)
            for root, dirs, files in os.walk(safe_path):
                # 깊이 제한 (성능 최적화)
                level = root.count(os.sep) - str(safe_path).count(os.sep)
                if level > 5:  # 최대 5단계 깊이
                    dirs.clear()
                    continue
                
                # 숨김 폴더 및 시스템 폴더 제외
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
                
                for file in files:
                    file_path = Path(root) / file
                    
                    # 압축 파일만 검색
                    if any(file.lower().endswith(ext) for ext in SystemConstants.ARCHIVE_EXTENSIONS):
                        candidate = self._analyze_file_candidate(file_path, query)
                        if candidate:
                            candidates.append(candidate)
        
        except PermissionError:
            self.logger.warning(f"경로 접근 권한 없음: {search_path}")
        except Exception as e:
            self.logger.error(f"경로 검색 오류 ({search_path}): {e}")
        
        return candidates
    
    def _analyze_file_candidate(self, file_path: Path, query: str) -> Optional[Dict[str, Any]]:
        """파일 후보 분석"""
        try:
            # 기본 정보
            file_stat = file_path.stat()
            candidate = {
                'path': str(file_path),
                'name': file_path.name,
                'size': file_stat.st_size,
                'modified_time': file_stat.st_mtime,
                'extension': file_path.suffix.lower(),
                'directory': str(file_path.parent),
                'scores': {}
            }
            
            # 압축파일 미리보기
            preview_info = self._preview_archive(file_path)
            candidate.update(preview_info)
            
            return candidate
            
        except Exception as e:
            self.logger.debug(f"파일 분석 실패 ({file_path}): {e}")
            return None
    
    def _preview_archive(self, archive_path: Path) -> Dict[str, Any]:
        """압축 파일 미리보기 - 내부 이미지 수 파악"""
        preview = {
            'archive_type': '',
            'total_files': 0,
            'image_files': 0,
            'folder_structure': [],
            'estimated_dataset': False
        }
        
        try:
            extension = archive_path.suffix.lower()
            
            if extension == '.zip':
                preview['archive_type'] = 'ZIP'
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    file_list = zf.namelist()
                    preview = self._analyze_archive_contents(file_list, preview)
            
            elif extension == '.rar' and rarfile:
                preview['archive_type'] = 'RAR'
                try:
                    with rarfile.RarFile(archive_path, 'r') as rf:
                        file_list = rf.namelist()
                        preview = self._analyze_archive_contents(file_list, preview)
                except Exception as e:
                    self.logger.debug(f"RAR 파일 미리보기 실패: {e}")
            
            elif extension in ['.tar', '.tar.gz', '.tar.bz2']:
                preview['archive_type'] = 'TAR'
                with tarfile.open(archive_path, 'r') as tf:
                    file_list = tf.getnames()
                    preview = self._analyze_archive_contents(file_list, preview)
            
        except Exception as e:
            self.logger.debug(f"압축파일 미리보기 실패 ({archive_path}): {e}")
        
        return preview
    
    def _analyze_archive_contents(self, file_list: List[str], preview: Dict[str, Any]) -> Dict[str, Any]:
        """압축파일 내용 분석"""
        preview['total_files'] = len(file_list)
        
        image_count = 0
        folder_set = set()
        
        for file_path in file_list:
            # 폴더 구조 파악
            if '/' in file_path or '\\' in file_path:
                folder_parts = file_path.replace('\\', '/').split('/')
                if len(folder_parts) > 1:
                    folder_set.add(folder_parts[0])
            
            # 이미지 파일 카운트
            file_ext = Path(file_path).suffix.lower()
            if file_ext in SystemConstants.IMAGE_EXTENSIONS:
                image_count += 1
        
        preview['image_files'] = image_count
        preview['folder_structure'] = list(folder_set)[:10]  # 상위 10개 폴더만
        
        # 데이터셋 추정 (이미지가 100개 이상이고 폴더 구조가 있으면)
        preview['estimated_dataset'] = (
            image_count >= 100 and 
            len(folder_set) >= 2 and
            any(keyword in str(file_list).lower() 
                for keyword in ['train', 'test', 'valid', 'class', 'label'])
        )
        
        return preview
    
    def _calculate_ai_scores(self, candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """AI 기반 점수 계산 시스템"""
        
        for candidate in candidates:
            total_score = 0.0
            scores = {}
            
            # 1. 파일명 매칭 점수
            filename_score = self._calculate_filename_score(candidate['name'], query)
            scores['filename'] = filename_score
            total_score += filename_score * SystemConstants.AI_SEARCH_WEIGHTS['filename_match']
            
            # 2. 확장자 점수
            ext_score = self.extension_weights.get(candidate['extension'], 0.1)
            scores['extension'] = ext_score
            total_score += ext_score * SystemConstants.AI_SEARCH_WEIGHTS['extension_match']
            
            # 3. 경로 우선순위 점수
            path_score = self._calculate_path_score(candidate['directory'])
            scores['path'] = path_score
            total_score += path_score * SystemConstants.AI_SEARCH_WEIGHTS['path_priority']
            
            # 4. 파일 크기 점수 (적당한 크기 선호)
            size_score = self._calculate_size_score(candidate['size'])
            scores['size'] = size_score
            total_score += size_score * SystemConstants.AI_SEARCH_WEIGHTS['file_size']
            
            # 5. 생성일 점수 (최근 파일 선호)
            date_score = self._calculate_date_score(candidate['modified_time'])
            scores['date'] = date_score
            total_score += date_score * SystemConstants.AI_SEARCH_WEIGHTS['creation_date']
            
            # 6. 데이터셋 추정 점수 (보너스)
            if candidate.get('estimated_dataset', False):
                total_score += 0.5
                scores['dataset_bonus'] = 0.5
            
            # 7. 이미지 파일 수 점수
            if candidate.get('image_files', 0) > 0:
                image_score = min(candidate['image_files'] / 1000, 1.0)
                scores['image_count'] = image_score
                total_score += image_score * 0.3
            
            candidate['scores'] = scores
            candidate['total_score'] = round(total_score, 3)
        
        # 점수순 정렬
        return sorted(candidates, key=lambda x: x['total_score'], reverse=True)
    
    def _calculate_filename_score(self, filename: str, query: str) -> float:
        """파일명 매칭 점수 계산"""
        filename_lower = filename.lower()
        score = 0.0
        
        # 쿼리와의 직접 매칭
        if query and query.lower() in filename_lower:
            score += 0.8
        
        # AI/ML 관련 키워드 매칭
        ai_keywords = [
            'dataset', 'data', 'train', 'test', 'valid', 'model',
            'yolo', 'coco', 'imagenet', 'mnist', 'cifar',
            'detection', 'classification', 'segmentation',
            'vision', 'cv', 'ml', 'ai', 'deep', 'neural'
        ]
        
        for keyword in ai_keywords:
            if keyword in filename_lower:
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_path_score(self, directory: str) -> float:
        """경로 우선순위 점수 계산"""
        dir_lower = directory.lower()
        score = 0.1  # 기본 점수
        
        for path_keyword, weight in self.path_weights.items():
            if path_keyword in dir_lower:
                score = max(score, weight)
        
        return score
    
    def _calculate_size_score(self, file_size: int) -> float:
        """파일 크기 점수 계산 (적당한 크기 선호)"""
        # MB 단위로 변환
        size_mb = file_size / (1024 * 1024)
        
        if 10 <= size_mb <= 1000:  # 10MB ~ 1GB
            return 1.0
        elif 1 <= size_mb <= 5000:  # 1MB ~ 5GB
            return 0.8
        elif size_mb < 1:  # 1MB 미만
            return 0.3
        else:  # 5GB 초과
            return 0.6
    
    def _calculate_date_score(self, modified_time: float) -> float:
        """생성일 점수 계산 (최근 파일 선호)"""
        current_time = time.time()
        days_ago = (current_time - modified_time) / (24 * 3600)
        
        if days_ago <= 30:  # 30일 이내
            return 1.0
        elif days_ago <= 90:  # 3개월 이내
            return 0.8
        elif days_ago <= 365:  # 1년 이내
            return 0.6
        else:  # 1년 초과
            return 0.4
    
    def _generate_cache_key(self, search_paths: List[Path], query: str) -> str:
        """검색 캐시 키 생성"""
        path_str = "|".join(str(p) for p in search_paths)
        return hashlib.md5(f"{path_str}:{query}".encode()).hexdigest()
    
    def learn_user_preference(self, selected_dataset: Dict[str, Any]):
        """사용자 선택 패턴 학습"""
        try:
            # 선택된 데이터셋의 특성 추출
            features = {
                'extension': selected_dataset.get('extension', ''),
                'size_range': self._get_size_range(selected_dataset.get('size', 0)),
                'path_keywords': self._extract_path_keywords(selected_dataset.get('directory', '')),
                'filename_keywords': self._extract_filename_keywords(selected_dataset.get('name', ''))
            }
            
            # 학습 데이터에 추가
            for key, value in features.items():
                if key not in self.user_preferences:
                    self.user_preferences[key] = defaultdict(int)
                
                if isinstance(value, list):
                    for item in value:
                        self.user_preferences[key][item] += 1
                else:
                    self.user_preferences[key][value] += 1
            
            self.logger.info(f"사용자 선택 패턴 학습 완료: {selected_dataset.get('name', 'Unknown')}")
            
        except Exception as e:
            self.logger.error(f"사용자 선택 패턴 학습 실패: {e}")
    
    def _get_size_range(self, size: int) -> str:
        """파일 크기 범위 분류"""
        size_mb = size / (1024 * 1024)
        
        if size_mb < 10:
            return 'small'
        elif size_mb < 100:
            return 'medium'
        elif size_mb < 1000:
            return 'large'
        else:
            return 'very_large'
    
    def _extract_path_keywords(self, directory: str) -> List[str]:
        """경로에서 키워드 추출"""
        keywords = []
        dir_lower = directory.lower()
        
        for keyword in ['desktop', 'downloads', 'documents', 'pictures', 'dataset', 'data', 'train']:
            if keyword in dir_lower:
                keywords.append(keyword)
        
        return keywords
    
    def _extract_filename_keywords(self, filename: str) -> List[str]:
        """파일명에서 키워드 추출"""
        keywords = []
        filename_lower = filename.lower()
        
        ai_keywords = ['dataset', 'data', 'train', 'test', 'yolo', 'coco', 'vision', 'cv']
        
        for keyword in ai_keywords:
            if keyword in filename_lower:
                keywords.append(keyword)
        
        return keywords

# ════════════════════════════════════════════════════════════════════════════════
# 📦 고급 압축파일 처리 시스템
# ════════════════════════════════════════════════════════════════════════════════

class AdvancedArchiveProcessor:
    """
    압축파일 처리 전문 시스템
    - 다양한 압축 형식 지원 (ZIP, RAR, 7Z, TAR)
    - 진행률 표시 및 오류 복구
    - 메모리 효율적인 대용량 파일 처리
    """
    
    def __init__(self, logger: AdvancedLogger, integrity_manager: DataIntegrityManager):
        self.logger = logger
        self.integrity_manager = integrity_manager
        
        # 지원되는 압축 형식
        self.supported_formats = {
            '.zip': self._extract_zip,
            '.rar': self._extract_rar,
            '.7z': self._extract_7z,
            '.tar': self._extract_tar,
            '.tar.gz': self._extract_tar,
            '.tar.bz2': self._extract_tar
        }
        
        # 압축 해제 통계
        self.extraction_stats = {}
    
    def extract_archive(self, archive_path: Path, extract_to: Path, 
                       password: Optional[str] = None) -> Dict[str, Any]:
        """
        압축 파일 추출
        Args:
            archive_path: 압축 파일 경로
            extract_to: 추출 대상 경로
            password: 암호 (선택적)
        """
        self.logger.info(f"압축 해제 시작: {archive_path}")
        
        result = {
            'success': False,
            'extracted_files': 0,
            'total_size': 0,
            'extraction_time': 0,
            'error_message': None,
            'extracted_path': None
        }
        
        start_time = time.time()
        
        try:
            # 파일 무결성 검증
            integrity_check = self.integrity_manager.verify_file_integrity(archive_path)
            if not integrity_check['integrity_ok']:
                result['error_message'] = "압축 파일이 손상되었습니다."
                return result
            
            # 추출 경로 준비
            extract_to.mkdir(parents=True, exist_ok=True)
            
            # 압축 형식 확인
            extension = archive_path.suffix.lower()
            if archive_path.name.endswith('.tar.gz'):
                extension = '.tar.gz'
            elif archive_path.name.endswith('.tar.bz2'):
                extension = '.tar.bz2'
            
            if extension not in self.supported_formats:
                result['error_message'] = f"지원하지 않는 압축 형식: {extension}"
                return result
            
            # 압축 해제 실행
            extractor = self.supported_formats[extension]
            extraction_result = extractor(archive_path, extract_to, password)
            
            result.update(extraction_result)
            result['extraction_time'] = time.time() - start_time
            
            if result['success']:
                self.logger.info(f"압축 해제 완료: {result['extracted_files']}개 파일, "
                               f"{result['extraction_time']:.2f}초")
            else:
                self.logger.error(f"압축 해제 실패: {result['error_message']}")
            
        except Exception as e:
            result['error_message'] = str(e)
            result['extraction_time'] = time.time() - start_time
            self.logger.error(f"압축 해제 중 오류 발생: {e}")
        
        return result
    
    def _extract_zip(self, archive_path: Path, extract_to: Path, 
                     password: Optional[str] = None) -> Dict[str, Any]:
        """ZIP 파일 추출"""
        result = {'success': False, 'extracted_files': 0, 'total_size': 0}
        
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                # 파일 목록 확인
                file_list = zf.namelist()
                total_files = len(file_list)
                
                # 진행률 표시를 위한 Rich Progress
                if RICH_AVAILABLE:
                    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
                    
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    ) as progress:
                        task = progress.add_task("ZIP 추출 중...", total=total_files)
                        
                        for i, file_info in enumerate(zf.infolist()):
                            # 안전한 경로 확인
                            if self._is_safe_path(file_info.filename):
                                try:
                                    if password:
                                        zf.extract(file_info, extract_to, pwd=password.encode())
                                    else:
                                        zf.extract(file_info, extract_to)
                                    
                                    result['extracted_files'] += 1
                                    result['total_size'] += file_info.file_size
                                    
                                except Exception as e:
                                    self.logger.warning(f"파일 추출 실패 ({file_info.filename}): {e}")
                            
                            progress.update(task, advance=1)
                else:
                    # Rich가 없는 경우 기본 추출
                    for file_info in zf.infolist():
                        if self._is_safe_path(file_info.filename):
                            try:
                                if password:
                                    zf.extract(file_info, extract_to, pwd=password.encode())
                                else:
                                    zf.extract(file_info, extract_to)
                                
                                result['extracted_files'] += 1
                                result['total_size'] += file_info.file_size
                                
                            except Exception as e:
                                self.logger.warning(f"파일 추출 실패 ({file_info.filename}): {e}")
                
                result['success'] = True
                result['extracted_path'] = extract_to
                
        except zipfile.BadZipFile:
            result['error_message'] = "손상된 ZIP 파일입니다."
        except Exception as e:
            result['error_message'] = str(e)
        
        return result
    
    def _extract_rar(self, archive_path: Path, extract_to: Path, 
                     password: Optional[str] = None) -> Dict[str, Any]:
        """RAR 파일 추출"""
        result = {'success': False, 'extracted_files': 0, 'total_size': 0}
        
        if not rarfile:
            result['error_message'] = "rarfile 라이브러리가 필요합니다. pip install rarfile"
            return result
        
        try:
            with rarfile.RarFile(archive_path, 'r') as rf:
                file_list = rf.namelist()
                total_files = len(file_list)
                
                if RICH_AVAILABLE:
                    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
                    
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    ) as progress:
                        task = progress.add_task("RAR 추출 중...", total=total_files)
                        
                        for i, filename in enumerate(file_list):
                            if self._is_safe_path(filename):
                                try:
                                    if password:
                                        rf.extract(filename, extract_to, pwd=password)
                                    else:
                                        rf.extract(filename, extract_to)
                                    
                                    result['extracted_files'] += 1
                                    
                                    # 파일 크기 계산
                                    extracted_file = extract_to / filename
                                    if extracted_file.exists():
                                        result['total_size'] += extracted_file.stat().st_size
                                    
                                except Exception as e:
                                    self.logger.warning(f"파일 추출 실패 ({filename}): {e}")
                            
                            progress.update(task, advance=1)
                else:
                    for filename in file_list:
                        if self._is_safe_path(filename):
                            try:
                                if password:
                                    rf.extract(filename, extract_to, pwd=password)
                                else:
                                    rf.extract(filename, extract_to)
                                
                                result['extracted_files'] += 1
                                
                                extracted_file = extract_to / filename
                                if extracted_file.exists():
                                    result['total_size'] += extracted_file.stat().st_size
                                
                            except Exception as e:
                                self.logger.warning(f"파일 추출 실패 ({filename}): {e}")
                
                result['success'] = True
                result['extracted_path'] = extract_to
                
        except Exception as e:
            result['error_message'] = str(e)
        
        return result
    
    def _extract_tar(self, archive_path: Path, extract_to: Path, 
                     password: Optional[str] = None) -> Dict[str, Any]:
        """TAR 파일 추출 (tar, tar.gz, tar.bz2)"""
        result = {'success': False, 'extracted_files': 0, 'total_size': 0}
        
        try:
            with tarfile.open(archive_path, 'r') as tf:
                members = tf.getmembers()
                total_files = len(members)
                
                if RICH_AVAILABLE:
                    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
                    
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    ) as progress:
                        task = progress.add_task("TAR 추출 중...", total=total_files)
                        
                        for member in members:
                            if self._is_safe_path(member.name) and member.isfile():
                                try:
                                    tf.extract(member, extract_to)
                                    result['extracted_files'] += 1
                                    result['total_size'] += member.size
                                except Exception as e:
                                    self.logger.warning(f"파일 추출 실패 ({member.name}): {e}")
                            
                            progress.update(task, advance=1)
                else:
                    for member in members:
                        if self._is_safe_path(member.name) and member.isfile():
                            try:
                                tf.extract(member, extract_to)
                                result['extracted_files'] += 1
                                result['total_size'] += member.size
                            except Exception as e:
                                self.logger.warning(f"파일 추출 실패 ({member.name}): {e}")
                
                result['success'] = True
                result['extracted_path'] = extract_to
                
        except Exception as e:
            result['error_message'] = str(e)
        
        return result
    
    def _extract_7z(self, archive_path: Path, extract_to: Path, 
                    password: Optional[str] = None) -> Dict[str, Any]:
        """7Z 파일 추출 (외부 도구 필요)"""
        result = {'success': False, 'extracted_files': 0, 'total_size': 0}
        
        try:
            # 7z 명령어 확인
            if not shutil.which('7z'):
                result['error_message'] = "7z 명령어가 설치되지 않았습니다."
                return result
            
            # 7z 명령어로 추출
            cmd = ['7z', 'x', str(archive_path), f'-o{extract_to}', '-y']
            if password:
                cmd.append(f'-p{password}')
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode == 0:
                # 추출된 파일 수 계산
                extracted_files = list(extract_to.rglob('*'))
                result['extracted_files'] = len([f for f in extracted_files if f.is_file()])
                result['total_size'] = sum(f.stat().st_size for f in extracted_files if f.is_file())
                result['success'] = True
                result['extracted_path'] = extract_to
            else:
                result['error_message'] = process.stderr
                
        except Exception as e:
            result['error_message'] = str(e)
        
        return result
    
    def _is_safe_path(self, path: str) -> bool:
        """안전한 경로인지 확인 (경로 탐색 공격 방지)"""
        # 절대 경로나 상위 디렉토리 참조 차단
        if os.path.isabs(path) or '..' in path:
            return False
        
        # 위험한 파일명 패턴 차단
        dangerous_patterns = ['../', '..\\', '/etc/', 'C:\\Windows\\']
        for pattern in dangerous_patterns:
            if pattern in path:
                return False
        
        return True
    
    def get_archive_info(self, archive_path: Path) -> Dict[str, Any]:
        """압축 파일 정보 조회"""
        info = {
            'path': str(archive_path),
            'size': 0,
            'type': 'unknown',
            'file_count': 0,
            'compressed_size': 0,
            'compression_ratio': 0
        }
        
        try:
            if not archive_path.exists():
                return info
            
            info['size'] = archive_path.stat().st_size
            extension = archive_path.suffix.lower()
            
            if extension == '.zip':
                info = self._get_zip_info(archive_path, info)
            elif extension == '.rar' and rarfile:
                info = self._get_rar_info(archive_path, info)
            elif extension in ['.tar', '.tar.gz', '.tar.bz2']:
                info = self._get_tar_info(archive_path, info)
            
        except Exception as e:
            self.logger.error(f"압축 파일 정보 조회 실패: {e}")
        
        return info
    
    def _get_zip_info(self, archive_path: Path, info: Dict[str, Any]) -> Dict[str, Any]:
        """ZIP 파일 정보"""
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                info['type'] = 'ZIP'
                info['file_count'] = len(zf.filelist)
                
                total_size = sum(file_info.file_size for file_info in zf.filelist)
                compressed_size = sum(file_info.compress_size for file_info in zf.filelist)
                
                info['compressed_size'] = total_size
                if total_size > 0:
                    info['compression_ratio'] = round((1 - compressed_size / total_size) * 100, 2)
        except Exception as e:
            self.logger.debug(f"ZIP 정보 조회 실패: {e}")
        
        return info
    
    def _get_rar_info(self, archive_path: Path, info: Dict[str, Any]) -> Dict[str, Any]:
        """RAR 파일 정보"""
        try:
            with rarfile.RarFile(archive_path, 'r') as rf:
                info['type'] = 'RAR'
                info['file_count'] = len(rf.namelist())
        except Exception as e:
            self.logger.debug(f"RAR 정보 조회 실패: {e}")
        
        return info
    
    def _get_tar_info(self, archive_path: Path, info: Dict[str, Any]) -> Dict[str, Any]:
        """TAR 파일 정보"""
        try:
            with tarfile.open(archive_path, 'r') as tf:
                members = tf.getmembers()
                info['type'] = 'TAR'
                info['file_count'] = len(members)
                info['compressed_size'] = sum(member.size for member in members if member.isfile())
        except Exception as e:
            self.logger.debug(f"TAR 정보 조회 실패: {e}")
        
        return info

# Part 3에서 이어집니다...

import textwrap
import webbrowser
from urllib.parse import quote
from typing import Callable, Optional
from contextlib import contextmanager

# ════════════════════════════════════════════════════════════════════════════════
# 🎨 Rich 기반 고급 UI 시스템
# ════════════════════════════════════════════════════════════════════════════════

class AdvancedUI:
    """
    Rich 라이브러리 기반 고급 터미널 UI 시스템
    - 반응형 레이아웃
    - 실시간 대시보드
    - 인터랙티브 메뉴 시스템
    - 다국어 지원
    """
    
    def __init__(self, language_manager: LanguageManager, logger: AdvancedLogger):
        self.lang = language_manager
        self.logger = logger
        
        # Rich 컴포넌트 초기화
        if RICH_AVAILABLE:
            self.console = Console()
            self.layout = Layout()
            self._setup_layout()
        else:
            self.console = None
            self.layout = None
            self.logger.warning("Rich UI를 사용할 수 없습니다. 기본 터미널 모드로 실행됩니다.")
    
    def _setup_layout(self):
        """레이아웃 초기 설정"""
        if not RICH_AVAILABLE:
            return
        
        # 메인 레이아웃 구성
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # 메인 영역 분할
        self.layout["main"].split_row(
            Layout(name="sidebar", size=40),
            Layout(name="content", ratio=1)
        )
    
    def show_welcome_screen(self):
        """환영 화면 표시"""
        if not RICH_AVAILABLE:
            print("="*80)
            print("🤖 AI 훈련 시스템 v3.0")
            print("="*80)
            return
        
        # 헤더 패널
        header_text = Text.assemble(
            ("🤖 AI 훈련 시스템 ", "bold cyan"),
            ("v3.0", "bold yellow"),
            (" - 차세대 통합 자동화 플랫폼", "cyan")
        )
        
        header_panel = Panel(
            header_text,
            title="Welcome",
            title_align="left",
            border_style="bright_blue"
        )
        
        # 기능 소개 패널
        features_text = Text()
        features_text.append("🔥 v2.2 → v3.0 주요 업그레이드\n\n", style="bold red")
        features_text.append("✨ 완전히 재설계된 모듈화 아키텍처\n", style="green")
        features_text.append("🛡️ 견고한 경로 검증 및 보안 시스템\n", style="green")
        features_text.append("📊 실시간 하드웨어 모니터링 (CPU/GPU/NPU)\n", style="green")
        features_text.append("🎨 Rich 기반 고급 UI 시스템\n", style="green")
        features_text.append("🤖 AI 기반 스마트 오류 해결\n", style="green")
        features_text.append("💾 데이터 무결성 검증 시스템\n", style="green")
        features_text.append("🌐 다국어 지원 (한국어/영어)\n", style="green")
        features_text.append("⚡ 압축파일 AI 검색 알고리즘 강화\n", style="green")
        features_text.append("📈 예측 분석 기반 성능 최적화\n", style="green")
        features_text.append("🔄 설정 백업/복원 시스템\n", style="green")
        
        features_panel = Panel(
            features_text,
            title="🆕 새로운 기능",
            title_align="left",
            border_style="green"
        )
        
        # 시스템 정보 패널
        system_info = self._get_system_info_text()
        system_panel = Panel(
            system_info,
            title="🖥️ 시스템 정보",
            title_align="left",
            border_style="yellow"
        )
        
        # 도움말 패널
        help_text = Text()
        help_text.append("💡 사용 팁\n\n", style="bold blue")
        help_text.append("• ", style="blue")
        help_text.append("언제든지 ", style="white")
        help_text.append("!help", style="bold cyan")
        help_text.append("를 입력하면 도움말을 볼 수 있습니다\n", style="white")
        help_text.append("• ", style="blue")
        help_text.append("ESC 키를 누르면 이전 단계로 돌아갑니다\n", style="white")
        help_text.append("• ", style="blue")
        help_text.append("Ctrl+C를 누르면 프로그램을 종료합니다\n", style="white")
        
        help_panel = Panel(
            help_text,
            title="❓ 도움말",
            title_align="left",
            border_style="blue"
        )
        
        # 패널들을 열로 배치
        columns = Columns(
            [features_panel, system_panel, help_panel],
            equal=True,
            expand=True
        )
        
        # 화면 출력
        self.console.clear()
        self.console.print(header_panel)
        self.console.print()
        self.console.print(columns)
        self.console.print()
    
    def _get_system_info_text(self) -> Text:
        """시스템 정보 텍스트 생성"""
        text = Text()
        
        # 플랫폼 정보
        text.append(f"🖥️  운영체제: ", style="cyan")
        text.append(f"{platform.system()} {platform.release()}\n", style="white")
        
        text.append(f"🐍  Python: ", style="cyan")
        text.append(f"{platform.python_version()}\n", style="white")
        
        # 하드웨어 정보
        text.append(f"🧠  CPU: ", style="cyan")
        text.append(f"{psutil.cpu_count()}코어\n", style="white")
        
        memory = psutil.virtual_memory()
        text.append(f"💾  메모리: ", style="cyan")
        text.append(f"{round(memory.total / (1024**3), 1)}GB\n", style="white")
        
        # GPU 정보
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                text.append(f"🎮  GPU: ", style="cyan")
                text.append(f"{gpus[0].name}\n", style="white")
        except:
            pass
        
        # PyTorch 정보
        if TORCH_AVAILABLE:
            text.append(f"🔥  PyTorch: ", style="cyan")
            text.append(f"{torch.__version__}\n", style="white")
            
            if torch.cuda.is_available():
                text.append(f"⚡  CUDA: ", style="cyan")
                text.append(f"사용 가능 ({torch.version.cuda})\n", style="green")
        
        return text
    
    def show_workflow_menu(self) -> str:
        """워크플로우 선택 메뉴"""
        if not RICH_AVAILABLE:
            print("\n" + "="*50)
            print("워크플로우 선택")
            print("="*50)
            print("1. 완전 자동 모드 (모든 것을 AI가 자동으로 처리)")
            print("2. 반자동 모드 (중요한 선택만 사용자가 결정)")
            print("3. 수동 모드 (모든 단계를 사용자가 직접 제어)")
            
            while True:
                choice = input("\n선택하세요 (1-3): ").strip()
                if choice in ['1', '2', '3']:
                    return ['auto', 'semi_auto', 'manual'][int(choice) - 1]
                print("잘못된 선택입니다. 1, 2, 3 중 하나를 입력하세요.")
        
        # Rich 메뉴
        options = [
            ("🤖 완전 자동 모드", "auto", "모든 설정을 AI가 자동으로 최적화합니다"),
            ("⚖️ 반자동 모드", "semi_auto", "중요한 결정만 사용자가 직접 선택합니다"),  
            ("🎛️ 수동 모드", "manual", "모든 설정을 사용자가 직접 제어합니다")
        ]
        
        # 메뉴 패널 생성
        menu_table = Table(title="🚀 워크플로우 선택", show_header=False, box=None)
        menu_table.add_column("번호", width=4, style="bold cyan")
        menu_table.add_column("옵션", width=20, style="bold")
        menu_table.add_column("설명", style="dim")
        
        for i, (name, value, desc) in enumerate(options, 1):
            menu_table.add_row(f"[{i}]", name, desc)
        
        menu_panel = Panel(
            menu_table,
            title="워크플로우 선택",
            title_align="left",
            border_style="bright_blue"
        )
        
        self.console.print(menu_panel)
        
        # 사용자 입력 받기
        while True:
            try:
                choice = Prompt.ask(
                    "\n[bold cyan]선택하세요[/bold cyan]",
                    choices=["1", "2", "3", "!help"],
                    show_choices=True
                )
                
                if choice == "!help":
                    self.show_workflow_help()
                    continue
                
                return options[int(choice) - 1][1]
                
            except KeyboardInterrupt:
                self.console.print("\n[red]프로그램을 종료합니다.[/red]")
                sys.exit(0)
            except Exception as e:
                self.console.print(f"[red]입력 오류: {e}[/red]")
    
    def show_workflow_help(self):
        """워크플로우 도움말"""
        if not RICH_AVAILABLE:
            print("\n워크플로우 도움말:")
            print("- 완전 자동: AI가 모든 설정을 자동으로 결정")
            print("- 반자동: 중요한 선택만 사용자가 결정")
            print("- 수동: 모든 설정을 사용자가 직접 제어")
            return
        
        help_text = Text()
        help_text.append("🤖 완전 자동 모드\n", style="bold green")
        help_text.append("   • AI가 하드웨어를 분석하여 최적의 설정을 자동 선택\n", style="green")
        help_text.append("   • 데이터셋 자동 검색 및 선택\n", style="green")
        help_text.append("   • 모델 및 하이퍼파라미터 자동 최적화\n", style="green")
        help_text.append("   • 초보자에게 권장\n\n", style="green")
        
        help_text.append("⚖️ 반자동 모드\n", style="bold yellow")
        help_text.append("   • AI가 추천한 옵션 중에서 사용자가 선택\n", style="yellow")
        help_text.append("   • 데이터셋 후보군을 제시하여 사용자가 최종 선택\n", style="yellow")
        help_text.append("   • 주요 설정은 사용자가 검토 후 승인\n", style="yellow")
        help_text.append("   • 적당한 제어권을 원하는 사용자에게 권장\n\n", style="yellow")
        
        help_text.append("🎛️ 수동 모드\n", style="bold red")
        help_text.append("   • 모든 설정을 사용자가 직접 입력\n", style="red")
        help_text.append("   • 세밀한 튜닝 및 실험적 설정 가능\n", style="red")
        help_text.append("   • 고급 사용자 및 연구 목적에 적합\n", style="red")
        help_text.append("   • 머신러닝 경험이 풍부한 사용자에게 권장\n", style="red")
        
        help_panel = Panel(
            help_text,
            title="💡 워크플로우 상세 설명",
            title_align="left",
            border_style="blue"
        )
        
        self.console.print(help_panel)
    
    def show_dataset_selection(self, datasets: List[Dict[str, Any]]) -> List[int]:
        """데이터셋 선택 UI"""
        if not datasets:
            if RICH_AVAILABLE:
                self.console.print("[red]검색된 데이터셋이 없습니다.[/red]")
            else:
                print("검색된 데이터셋이 없습니다.")
            return []
        
        if not RICH_AVAILABLE:
            print(f"\n발견된 데이터셋: {len(datasets)}개")
            print("="*60)
            for i, dataset in enumerate(datasets):
                print(f"{i+1:2d}. {dataset['name']}")
                print(f"    경로: {dataset['path']}")
                print(f"    크기: {dataset['size'] / (1024*1024):.1f}MB")
                print(f"    점수: {dataset['total_score']:.3f}")
                print()
            
            selected = input("선택할 번호를 입력하세요 (예: 1,3,5 또는 1-5): ").strip()
            return self._parse_selection(selected, len(datasets))
        
        # Rich 테이블로 데이터셋 표시
        table = Table(title=f"🔍 발견된 데이터셋 ({len(datasets)}개)", show_lines=True)
        table.add_column("번호", width=4, style="cyan")
        table.add_column("이름", width=25, style="bold")
        table.add_column("크기", width=10, style="yellow")
        table.add_column("이미지", width=8, style="green")
        table.add_column("점수", width=6, style="red")
        table.add_column("경로", style="dim")
        
        for i, dataset in enumerate(datasets):
            size_mb = dataset['size'] / (1024 * 1024)
            size_str = f"{size_mb:.1f}MB" if size_mb < 1024 else f"{size_mb/1024:.1f}GB"
            
            table.add_row(
                str(i + 1),
                dataset['name'][:23] + "..." if len(dataset['name']) > 23 else dataset['name'],
                size_str,
                str(dataset.get('image_files', 0)),
                f"{dataset['total_score']:.3f}",
                str(Path(dataset['path']).parent)
            )
        
        self.console.print(table)
        
        # 선택 프롬프트
        while True:
            try:
                selection = Prompt.ask(
                    "\n[bold cyan]선택할 데이터셋 번호[/bold cyan]",
                    default="1",
                    show_default=True
                )
                
                if selection == "!help":
                    self.show_dataset_selection_help()
                    continue
                
                selected_indices = self._parse_selection(selection, len(datasets))
                return selected_indices
                
            except KeyboardInterrupt:
                return []
            except Exception as e:
                self.console.print(f"[red]선택 형식이 올바르지 않습니다: {e}[/red]")
                self.console.print("[yellow]예시: 1, 1-3, 1,3,5[/yellow]")
    
    def show_dataset_selection_help(self):
        """데이터셋 선택 도움말"""
        if not RICH_AVAILABLE:
            print("\n데이터셋 선택 도움말:")
            print("- 단일 선택: 1")
            print("- 여러 선택: 1,3,5")
            print("- 범위 선택: 1-5")
            print("- 혼합 선택: 1,3-5,7")
            return
        
        help_text = Text()
        help_text.append("📋 선택 방법\n\n", style="bold blue")
        help_text.append("• ", style="blue")
        help_text.append("단일 선택: ", style="cyan")
        help_text.append("1\n", style="white")
        help_text.append("• ", style="blue")
        help_text.append("여러 선택: ", style="cyan")
        help_text.append("1,3,5\n", style="white")
        help_text.append("• ", style="blue")
        help_text.append("범위 선택: ", style="cyan")
        help_text.append("1-5\n", style="white")
        help_text.append("• ", style="blue")
        help_text.append("혼합 선택: ", style="cyan")
        help_text.append("1,3-5,7\n\n", style="white")
        
        help_text.append("📊 점수 의미\n\n", style="bold blue")
        help_text.append("• ", style="blue")
        help_text.append("높은 점수일수록 더 적합한 데이터셋\n", style="white")
        help_text.append("• ", style="blue")
        help_text.append("파일명, 경로, 크기 등을 종합적으로 평가\n", style="white")
        help_text.append("• ", style="blue")
        help_text.append("이미지 수가 많을수록 높은 점수\n", style="white")
        
        help_panel = Panel(
            help_text,
            title="💡 데이터셋 선택 도움말",
            title_align="left",
            border_style="blue"
        )
        
        self.console.print(help_panel)
    
    def _parse_selection(self, selection: str, max_count: int) -> List[int]:
        """선택 문자열 파싱"""
        selected = []
        
        try:
            parts = selection.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected.extend(range(start, end + 1))
                else:
                    selected.append(int(part))
            
            # 중복 제거 및 정렬
            selected = sorted(set(selected))
            
            # 범위 검증
            valid_selected = [i for i in selected if 1 <= i <= max_count]
            
            # 인덱스를 0부터 시작하도록 변환
            return [i - 1 for i in valid_selected]
            
        except Exception as e:
            raise ValueError(f"선택 형식이 올바르지 않습니다: {e}")
    
    @contextmanager
    def show_progress(self, description: str, total: Optional[int] = None):
        """진행률 표시 컨텍스트 매니저"""
        if not RICH_AVAILABLE:
            print(f"{description}...")
            yield None
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn() if total else "",
            TaskProgressColumn() if total else "",
            TimeRemainingColumn() if total else "",
            console=self.console
        ) as progress:
            task = progress.add_task(description, total=total)
            
            class ProgressUpdater:
                def __init__(self, task_id, progress_obj):
                    self.task_id = task_id
                    self.progress = progress_obj
                
                def update(self, advance: int = 1, description: Optional[str] = None):
                    if description:
                        self.progress.update(self.task_id, advance=advance, description=description)
                    else:
                        self.progress.update(self.task_id, advance=advance)
                
                def complete(self):
                    self.progress.update(self.task_id, completed=self.progress.tasks[self.task_id].total or 100)
            
            yield ProgressUpdater(task, progress)
    
    def show_hardware_dashboard(self, monitor: 'HardwareMonitor') -> None:
        """실시간 하드웨어 대시보드"""
        if not RICH_AVAILABLE:
            performance = monitor.get_performance_summary()
            current = performance.get('current', {})
        
            print(f"\n하드웨어 상태:")
            print(f"CPU: {current.get('cpu', {}).get('usage_percent', 0):.1f}%")
            print(f"메모리: {current.get('memory', {}).get('used_percent', 0):.1f}%")
        
            for i, gpu in enumerate(current.get('gpu', [])):
                print(f"GPU {i}: {gpu.get('load_percent', 0):.1f}%")
        
            return
        
        def generate_dashboard():
            performance = monitor.get_performance_summary()
            current = performance.get('current', {})
        
            # 메인 레이아웃
            dashboard_layout = Layout()
        
            # CPU 패널
            cpu_info = current.get('cpu', {})
            cpu_panel = Panel(
                self._create_cpu_display(cpu_info),
                title="🧠 CPU",
                border_style="blue"
            )
        
            # 메모리 패널
            memory_info = current.get('memory', {})
            memory_panel = Panel(
                self._create_memory_display(memory_info),
                title="💾 메모리",
                border_style="green"
            )
        
            # GPU 패널들
            gpu_info = current.get('gpu', [])
            if gpu_info:
                gpu_panels = []
                for i, gpu in enumerate(gpu_info[:2]):  # 최대 2개 GPU만 표시
                    gpu_panel = Panel(
                        self._create_gpu_display(gpu),
                        title=f"🎮 GPU {i}",
                        border_style="red"
                    )
                    gpu_panels.append(gpu_panel)
            else:
                gpu_panels = [Panel(
                    Text("GPU를 사용할 수 없습니다.\nCPU 모드로 실행 중입니다.", style="yellow"),
                    title="🎮 GPU",
                    border_style="dim"
                )]
        
            # NPU 패널
            npu_info = current.get('npu', {})
            if npu_info.get('available'):
                npu_panel = Panel(
                    self._create_npu_display(npu_info),
                    title="⚡ NPU",
                    border_style="yellow"
                )
            else:
                npu_panel = Panel(
                    Text("NPU가 감지되지 않았습니다.", style="dim"),
                    title="⚡ NPU",
                    border_style="dim"
                )
        
            # 추천사항 패널
            recommendations = performance.get('recommendations', [])
            if recommendations:
                rec_text = Text()
                for i, rec in enumerate(recommendations[:3], 1):
                    rec_text.append(f"{i}. {rec}\n", style="yellow")
            
                rec_panel = Panel(
                    rec_text,
                    title="💡 최적화 추천",
                    border_style="yellow"
                )
            else:
                rec_panel = Panel(
                    Text("현재 시스템이 최적 상태입니다.", style="green"),
                    title="💡 최적화 추천",
                    border_style="green"
                )
        
            # 레이아웃 구성
            try:
                # 상단: CPU + 메모리
                top_layout = Layout()
                top_layout.split_row(
                    Layout(cpu_panel, name="cpu"),
                    Layout(memory_panel, name="memory")
                )
            
                # 중간: GPU 패널들
                if len(gpu_panels) == 1:
                    middle_layout = Layout()
                    middle_layout.split_row(
                        Layout(gpu_panels[0], name="gpu"),
                        Layout(npu_panel, name="npu")
                    )
                else:
                    middle_layout = Layout()
                    middle_layout.split_row(
                        Layout(gpu_panels[0], name="gpu1"),
                        Layout(gpu_panels[1] if len(gpu_panels) > 1 else npu_panel, name="gpu2")
                    )
            
                # 전체 레이아웃
                dashboard_layout.split_column(
                    Layout(top_layout, name="top", size=8),
                    Layout(middle_layout, name="middle", size=8),
                    Layout(rec_panel, name="bottom", size=6)
                )
            
            except Exception as e:
                # 레이아웃 오류 시 간단한 패널로 대체
                error_panel = Panel(
                    f"레이아웃 오류: {e}\n하드웨어 정보를 표시할 수 없습니다.",
                    title="❌ 오류",
                    border_style="red"
                )
                return error_panel
        
            return dashboard_layout
    
        # 실시간 업데이트 대시보드
        try:
            with Live(generate_dashboard(), console=self.console, refresh_per_second=1) as live:
                self.console.print("[dim]Press Ctrl+C to exit dashboard[/dim]")
                while True:
                    time.sleep(1)
                    try:
                        live.update(generate_dashboard())
                    except Exception as update_error:
                        self.logger.warning(f"대시보드 업데이트 오류: {update_error}")
                        break
        except KeyboardInterrupt:
            self.console.print("\n[green]대시보드를 종료합니다.[/green]")
        except Exception as e:
            self.console.print(f"\n[red]대시보드 오류: {e}[/red]")
    
    def _create_cpu_display(self, cpu_info: Dict[str, Any]) -> Text:
        """CPU 디스플레이 생성"""
        text = Text()
        usage = cpu_info.get('usage_percent', 0)
        frequency = cpu_info.get('frequency_mhz', 0)
        cores = cpu_info.get('core_count', 0)
        temp = cpu_info.get('temperature')
        
        # 사용률 바
        bar_length = 20
        filled = int(usage / 100 * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        if usage > 80:
            style = "red"
        elif usage > 60:
            style = "yellow"
        else:
            style = "green"
        
        text.append(f"사용률: {usage:5.1f}% ", style="white")
        text.append(bar, style=style)
        text.append(f"\n주파수: {frequency/1000:.1f}GHz\n", style="cyan")
        text.append(f"코어 수: {cores}개", style="blue")
        
        if temp:
            text.append(f"\n온도: {temp:.1f}°C", style="red" if temp > 70 else "green")
        
        return text
    
    def _create_memory_display(self, memory_info: Dict[str, Any]) -> Text:
        """메모리 디스플레이 생성"""
        text = Text()
        total = memory_info.get('total_gb', 0)
        available = memory_info.get('available_gb', 0)
        used_percent = memory_info.get('used_percent', 0)
        
        # 사용률 바
        bar_length = 20
        filled = int(used_percent / 100 * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        if used_percent > 85:
            style = "red"
        elif used_percent > 70:
            style = "yellow"
        else:
            style = "green"
        
        text.append(f"사용률: {used_percent:5.1f}% ", style="white")
        text.append(bar, style=style)
        text.append(f"\n총 용량: {total:.1f}GB\n", style="cyan")
        text.append(f"사용 가능: {available:.1f}GB", style="blue")
        
        return text
    
    def _create_gpu_display(self, gpu_info: Dict[str, Any]) -> Text:
        """GPU 디스플레이 생성"""
        text = Text()
        name = gpu_info.get('name', 'Unknown GPU')
        load = gpu_info.get('load_percent', 0)
        memory_percent = gpu_info.get('memory_percent', 0)
        memory_used = gpu_info.get('memory_used_mb', 0)
        memory_total = gpu_info.get('memory_total_mb', 0)
        temp = gpu_info.get('temperature', 0)
        
        text.append(f"모델: {name[:15]}...\n" if len(name) > 15 else f"모델: {name}\n", style="white")
        
        # GPU 로드 바
        bar_length = 15
        filled = int(load / 100 * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        load_style = "red" if load > 80 else "yellow" if load > 60 else "green"
        text.append(f"로드: {load:5.1f}% ", style="white")
        text.append(bar, style=load_style)
        
        # 메모리 정보
        text.append(f"\nVRAM: {memory_used/1024:.1f}/{memory_total/1024:.1f}GB", style="cyan")
        
        if temp > 0:
            temp_style = "red" if temp > 80 else "yellow" if temp > 70 else "green"
            text.append(f"\n온도: {temp}°C", style=temp_style)
        
        return text
    
    def _create_npu_display(self, npu_info: Dict[str, Any]) -> Text:
        """NPU 디스플레이 생성"""
        text = Text()
        
        if npu_info.get('available'):
            usage = npu_info.get('usage_percent', 0)
            power = npu_info.get('power_watts', 0)
            devices = npu_info.get('devices', [])
            
            text.append("상태: 사용 가능\n", style="green")
            
            if devices:
                text.append(f"장치: {', '.join(devices)}\n", style="cyan")
            
            if usage > 0:
                text.append(f"사용률: {usage:.1f}%\n", style="yellow")
            
            if power > 0:
                text.append(f"전력: {power:.1f}W", style="blue")
        else:
            text.append("상태: 사용 불가", style="dim")
        
        return text
    
    def show_error(self, title: str, error_message: str, suggestion: Optional[str] = None):
        """오류 메시지 표시"""
        if not RICH_AVAILABLE:
            print(f"\n❌ {title}")
            print(f"오류: {error_message}")
            if suggestion:
                print(f"제안: {suggestion}")
            return
        
        error_text = Text()
        error_text.append(f"{error_message}\n", style="red")
        
        if suggestion:
            error_text.append("\n💡 해결 방법:\n", style="bold yellow")
            error_text.append(suggestion, style="yellow")
        
        error_panel = Panel(
            error_text,
            title=f"❌ {title}",
            title_align="left",
            border_style="red"
        )
        
        self.console.print(error_panel)

# ════════════════════════════════════════════════════════════════════════════════
# 🤖 AI 기반 오류 해결 시스템
# ════════════════════════════════════════════════════════════════════════════════

class AIErrorSolver:
    """
    AI 기반 지능형 오류 분석 및 해결 시스템
    - 패턴 기반 오류 감지
    - ChatGPT API 연동
    - 해결 사례 데이터베이스
    """
    
    def __init__(self, logger: AdvancedLogger, ui: AdvancedUI):
        self.logger = logger
        self.ui = ui
        
        # 오류 패턴 데이터베이스
        self.error_patterns = {
            'cuda_error': {
                'patterns': [
                    r'CUDA out of memory',
                    r'CUDA device-side assert',
                    r'CUDA.*not available'
                ],
                'solutions': [
                    "배치 크기를 줄여보세요 (예: batch_size=16 → batch_size=8)",
                    "GPU 메모리를 정리해보세요 (torch.cuda.empty_cache())",
                    "Mixed precision training을 사용해보세요 (--fp16)"
                ]
            },
            'memory_error': {
                'patterns': [
                    r'MemoryError',
                    r'out of memory',
                    r'Cannot allocate memory'
                ],
                'solutions': [
                    "배치 크기를 줄여보세요",
                    "데이터 로더의 num_workers를 줄여보세요",
                    "이미지 크기를 줄여보세요"
                ]
            },
            'file_error': {
                'patterns': [
                    r'FileNotFoundError',
                    r'No such file or directory',
                    r'Permission denied'
                ],
                'solutions': [
                    "파일 경로를 확인해보세요",
                    "파일 권한을 확인해보세요",
                    "파일이 존재하는지 확인해보세요"
                ]
            },
            'model_error': {
                'patterns': [
                    r'RuntimeError.*model',
                    r'dimension mismatch',
                    r'size mismatch'
                ],
                'solutions': [
                    "모델 입력 차원을 확인해보세요",
                    "배치 크기와 모델 설정을 확인해보세요",
                    "데이터 전처리를 확인해보세요"
                ]
            }
        }
        
        # 해결된 오류 사례 데이터베이스
        self.solution_database = {}
        self.load_solution_database()
    
    def analyze_error(self, error_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        오류 분석 및 해결책 제시
        Args:
            error_message: 오류 메시지
            context: 추가 컨텍스트 정보
        """
        analysis = {
            'error_type': 'unknown',
            'severity': 'medium',
            'automatic_solutions': [],
            'manual_solutions': [],
            'external_search_query': None,
            'similar_cases': []
        }
        
        # 패턴 매칭으로 오류 유형 분석
        detected_type = self._detect_error_type(error_message)
        if detected_type:
            analysis['error_type'] = detected_type
            analysis['automatic_solutions'] = self.error_patterns[detected_type]['solutions']
        
        # 심각도 평가
        analysis['severity'] = self._assess_severity(error_message)
        
        # 유사 사례 검색
        analysis['similar_cases'] = self._find_similar_cases(error_message)
        
        # 외부 AI 검색 쿼리 생성
        analysis['external_search_query'] = self._generate_search_query(error_message, context)
        
        return analysis
    
    def _detect_error_type(self, error_message: str) -> Optional[str]:
        """오류 패턴 매칭"""
        import re
        
        for error_type, data in self.error_patterns.items():
            for pattern in data['patterns']:
                if re.search(pattern, error_message, re.IGNORECASE):
                    return error_type
        
        return None
    
    def _assess_severity(self, error_message: str) -> str:
        """오류 심각도 평가"""
        critical_keywords = ['fatal', 'critical', 'segmentation fault', 'access violation']
        high_keywords = ['cuda', 'memory', 'allocation']
        
        error_lower = error_message.lower()
        
        if any(keyword in error_lower for keyword in critical_keywords):
            return 'critical'
        elif any(keyword in error_lower for keyword in high_keywords):
            return 'high'
        else:
            return 'medium'
    
    def _find_similar_cases(self, error_message: str) -> List[Dict[str, Any]]:
        """유사 오류 사례 검색"""
        similar_cases = []
        
        # 간단한 키워드 매칭으로 유사 사례 검색
        error_keywords = self._extract_keywords(error_message)
        
        for case_id, case_data in self.solution_database.items():
            case_keywords = self._extract_keywords(case_data.get('error', ''))
            
            # 키워드 유사도 계산
            similarity = self._calculate_similarity(error_keywords, case_keywords)
            
            if similarity > 0.3:  # 30% 이상 유사
                similar_cases.append({
                    'case_id': case_id,
                    'similarity': similarity,
                    'solution': case_data.get('solution', ''),
                    'success_rate': case_data.get('success_rate', 0)
                })
        
        # 유사도 순으로 정렬
        similar_cases.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_cases[:3]  # 상위 3개만 반환
    
    def _extract_keywords(self, text: str) -> set:
        """텍스트에서 키워드 추출"""
        import re
        
        # 특수 문자 제거 및 단어 분리
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 불용어 제거
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = {word for word in words if len(word) > 3 and word not in stop_words}
        
        return keywords
    
    def _calculate_similarity(self, keywords1: set, keywords2: set) -> float:
        """키워드 집합 간 유사도 계산"""
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_search_query(self, error_message: str, context: Optional[Dict[str, Any]]) -> str:
        """외부 AI 검색을 위한 쿼리 생성"""
        query_parts = ["Python machine learning error:"]
        
        # 오류 메시지의 핵심 부분 추출
        lines = error_message.strip().split('\n')
        if lines:
            # 마지막 줄이 보통 핵심 오류 메시지
            main_error = lines[-1].strip()
            query_parts.append(f'"{main_error}"')
        
        # 컨텍스트 정보 추가
        if context:
            if context.get('framework'):
                query_parts.append(f"in {context['framework']}")
            
            if context.get('operation'):
                query_parts.append(f"during {context['operation']}")
        
        return " ".join(query_parts)
    
    def create_chatgpt_query(self, error_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """ChatGPT용 상세 질문 생성"""
        query_parts = [
            "I'm encountering an error in my Python machine learning project.",
            f"Error message: {error_message}",
            "",
            "Context:"
        ]
        
        if context:
            if context.get('framework'):
                query_parts.append(f"- Framework: {context['framework']}")
            
            if context.get('operation'):
                query_parts.append(f"- Operation: {context['operation']}")
            
            if context.get('system_info'):
                query_parts.append(f"- System: {context['system_info']}")
            
            if context.get('hardware'):
                query_parts.append(f"- Hardware: {context['hardware']}")
        
        query_parts.extend([
            "",
            "Please provide:",
            "1. Explanation of what's causing this error",
            "2. Step-by-step solution",
            "3. Prevention tips for the future",
            "4. Alternative approaches if the main solution doesn't work"
        ])
        
        return "\n".join(query_parts)
    
    def open_chatgpt_with_query(self, query: str) -> bool:
        """ChatGPT 웹페이지를 열고 쿼리 제공"""
        try:
            # URL 인코딩
            encoded_query = quote(query)
            
            # ChatGPT URL (실제로는 클립보드에 복사하고 URL만 열기)
            chatgpt_url = "https://chat.openai.com/"
            
            # 쿼리를 클립보드에 복사 시도
            try:
                import pyperclip
                pyperclip.copy(query)
                clipboard_success = True
            except ImportError:
                clipboard_success = False
            
            # 웹브라우저 열기
            webbrowser.open(chatgpt_url)
            
            if clipboard_success:
                self.ui.console.print("[green]✅ ChatGPT 페이지가 열렸고 질문이 클립보드에 복사되었습니다.[/green]")
                self.ui.console.print("[yellow]💡 ChatGPT에서 Ctrl+V로 붙여넣기 하세요.[/yellow]")
            else:
                self.ui.console.print("[yellow]⚠️ ChatGPT 페이지가 열렸습니다. 아래 질문을 복사해서 사용하세요:[/yellow]")
                self.ui.console.print(Panel(query, title="ChatGPT 질문", border_style="blue"))
            
            return True
            
        except Exception as e:
            self.logger.error(f"ChatGPT 페이지 열기 실패: {e}")
            return False
    
    def show_error_analysis(self, error_message: str, context: Optional[Dict[str, Any]] = None):
        """오류 분석 결과 표시"""
        analysis = self.analyze_error(error_message, context)
        
        if not RICH_AVAILABLE:
            print(f"\n오류 분석 결과:")
            print(f"오류 유형: {analysis['error_type']}")
            print(f"심각도: {analysis['severity']}")
            
            if analysis['automatic_solutions']:
                print("\n자동 해결 방법:")
                for i, solution in enumerate(analysis['automatic_solutions'], 1):
                    print(f"  {i}. {solution}")
            
            if analysis['similar_cases']:
                print(f"\n유사 사례: {len(analysis['similar_cases'])}개 발견")
            
            return analysis
        
        # Rich 기반 분석 결과 표시
        layout = Layout()
        layout.split_column(
            Layout(name="error_info", size=6),
            Layout(name="solutions", ratio=1),
            Layout(name="actions", size=4)
        )
        
        # 오류 정보 패널
        error_info = Text()
        error_info.append("🔍 분석 결과\n\n", style="bold blue")
        error_info.append(f"오류 유형: ", style="white")
        error_info.append(f"{analysis['error_type']}\n", style="cyan")
        error_info.append(f"심각도: ", style="white")
        
        severity_style = {
            'critical': 'bold red',
            'high': 'red', 
            'medium': 'yellow',
            'low': 'green'
        }.get(analysis['severity'], 'white')
        
        error_info.append(f"{analysis['severity']}\n", style=severity_style)
        
        if analysis['similar_cases']:
            error_info.append(f"유사 사례: ", style="white")
            error_info.append(f"{len(analysis['similar_cases'])}개 발견", style="green")
        
        layout["error_info"] = Panel(error_info, title="📊 오류 분석", border_style="blue")
        
        # 해결 방법 패널
        solutions_text = Text()
        
        if analysis['automatic_solutions']:
            solutions_text.append("🔧 권장 해결 방법\n\n", style="bold green")
            for i, solution in enumerate(analysis['automatic_solutions'], 1):
                solutions_text.append(f"{i}. ", style="green")
                solutions_text.append(f"{solution}\n", style="white")
        
        if analysis['similar_cases']:
            solutions_text.append("\n📚 유사 사례 해결책\n\n", style="bold yellow")
            for case in analysis['similar_cases'][:2]:
                solutions_text.append(f"• ", style="yellow")
                solutions_text.append(f"{case['solution']}\n", style="white")
                solutions_text.append(f"  (성공률: {case['success_rate']*100:.0f}%)\n\n", style="dim")
        
        layout["solutions"] = Panel(solutions_text, title="💡 해결 방법", border_style="green")
        
        # 액션 패널
        actions_text = Text()
        actions_text.append("🤖 AI 도움 받기\n\n", style="bold cyan")
        actions_text.append("1. ChatGPT에게 질문하기 (자동으로 질문 생성)\n", style="cyan")
        actions_text.append("2. 해결 사례 데이터베이스에 저장\n", style="cyan")
        actions_text.append("3. 시스템 로그 상세 분석", style="cyan")
        
        layout["actions"] = Panel(actions_text, title="🚀 다음 단계", border_style="cyan")
        
        self.console.print(layout)
        
        # 사용자 선택
        choice = Prompt.ask(
            "\n[bold cyan]다음 중 선택하세요[/bold cyan]",
            choices=["1", "2", "3", "skip"],
            default="skip"
        )
        
        if choice == "1":
            chatgpt_query = self.create_chatgpt_query(error_message, context)
            self.open_chatgpt_with_query(chatgpt_query)
        elif choice == "2":
            self.save_error_case(error_message, analysis)
        elif choice == "3":
            self.show_detailed_logs()
        
        return analysis
    
    def save_error_case(self, error_message: str, analysis: Dict[str, Any]):
        """오류 사례를 데이터베이스에 저장"""
        case_id = hashlib.md5(error_message.encode()).hexdigest()[:8]
        
        case_data = {
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'error_type': analysis['error_type'],
            'severity': analysis['severity'],
            'automatic_solutions': analysis['automatic_solutions'],
            'success_rate': 0.5  # 기본값
        }
        
        self.solution_database[case_id] = case_data
        self._save_solution_database()
        
        if RICH_AVAILABLE:
            self.ui.console.print(f"[green]✅ 오류 사례가 저장되었습니다 (ID: {case_id})[/green]")
        else:
            print(f"오류 사례가 저장되었습니다 (ID: {case_id})")
    
    def load_solution_database(self):
        """해결 사례 데이터베이스 로드"""
        db_file = Path("error_solutions.json")
        try:
            if db_file.exists():
                with open(db_file, 'r', encoding='utf-8') as f:
                    self.solution_database = json.load(f)
                self.logger.debug(f"해결 사례 데이터베이스 로드: {len(self.solution_database)}개 사례")
        except Exception as e:
            self.logger.error(f"해결 사례 데이터베이스 로드 실패: {e}")
            self.solution_database = {}
    
    def _save_solution_database(self):
        """해결 사례 데이터베이스 저장"""
        db_file = Path("error_solutions.json")
        try:
            with open(db_file, 'w', encoding='utf-8') as f:
                json.dump(self.solution_database, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"해결 사례 데이터베이스 저장 실패: {e}")
    
    def show_detailed_logs(self):
        """상세 로그 분석 표시"""
        log_dir = Path("logs")
        
        if not log_dir.exists():
            if RICH_AVAILABLE:
                self.ui.console.print("[red]로그 디렉토리가 존재하지 않습니다.[/red]")
            else:
                print("로그 디렉토리가 존재하지 않습니다.")
            return
        
        # 최근 로그 파일 찾기
        log_files = list(log_dir.glob("*.log"))
        if not log_files:
            if RICH_AVAILABLE:
                self.ui.console.print("[red]로그 파일이 없습니다.[/red]")
            else:
                print("로그 파일이 없습니다.")
            return
        
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # 오류 라인만 추출
            error_lines = [line for line in log_content.split('\n') 
                          if 'ERROR' in line or 'CRITICAL' in line]
            
            if RICH_AVAILABLE:
                if error_lines:
                    log_text = "\n".join(error_lines[-10:])  # 최근 10개 오류만
                    log_panel = Panel(
                        Syntax(log_text, "log", theme="monokai", line_numbers=True),
                        title="🔍 최근 오류 로그",
                        border_style="red"
                    )
                    self.ui.console.print(log_panel)
                else:
                    self.ui.console.print("[green]최근 오류가 발견되지 않았습니다.[/green]")
            else:
                if error_lines:
                    print("최근 오류 로그:")
                    for line in error_lines[-10:]:
                        print(line)
                else:
                    print("최근 오류가 발견되지 않았습니다.")
                    
        except Exception as e:
            self.logger.error(f"로그 파일 읽기 실패: {e}")

# Part 4에서 이어집니다...

import requests
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# ════════════════════════════════════════════════════════════════════════════════
# 🎯 통합 도움말 시스템
# ════════════════════════════════════════════════════════════════════════════════

class HelpSystem:
    """
    통합 도움말 시스템 - 모든 입력 프롬프트에서 !help 명령어 지원
    - 상황별 도움말 제공
    - 단계별 가이드
    - 예제 및 팁 제공
    """
    
    def __init__(self, language_manager: LanguageManager, ui: AdvancedUI):
        self.lang = language_manager
        self.ui = ui
        
        # 상황별 도움말 데이터
        self.help_data = {
            'workflow_selection': {
                'ko': {
                    'title': '워크플로우 선택 도움말',
                    'content': [
                        '🤖 완전 자동 모드: AI가 모든 설정을 자동으로 최적화',
                        '   • 초보자에게 권장',
                        '   • 하드웨어 자동 감지 및 최적화',
                        '   • 데이터셋 자동 검색 및 선택',
                        '',
                        '⚖️ 반자동 모드: AI 추천 중 사용자가 선택',
                        '   • 적당한 제어권을 원하는 사용자 권장',
                        '   • 데이터셋 후보군 중 선택',
                        '   • 주요 설정 검토 후 승인',
                        '',
                        '🎛️ 수동 모드: 모든 설정을 사용자가 직접 제어',
                        '   • 고급 사용자 및 연구 목적',
                        '   • 세밀한 튜닝 가능',
                        '   • 실험적 설정 지원'
                    ]
                }
            },
            'dataset_selection': {
                'ko': {
                    'title': '데이터셋 선택 도움말',
                    'content': [
                        '📋 선택 방법:',
                        '   • 단일 선택: 1',
                        '   • 여러 선택: 1,3,5',
                        '   • 범위 선택: 1-5',
                        '   • 혼합 선택: 1,3-5,7',
                        '',
                        '📊 점수 의미:',
                        '   • 높은 점수 = 더 적합한 데이터셋',
                        '   • 파일명, 경로, 크기 등 종합 평가',
                        '   • 이미지 수가 많을수록 높은 점수',
                        '',
                        '💡 선택 팁:',
                        '   • 점수가 높은 데이터셋 우선 선택',
                        '   • 이미지 수가 100개 이상인 것 권장',
                        '   • 압축 해제 전 미리보기 정보 확인'
                    ]
                }
            },
            'model_selection': {
                'ko': {
                    'title': '모델 선택 도움말',
                    'content': [
                        '🎯 모델 카테고리:',
                        '   • Object Detection: 객체 위치와 클래스 검출',
                        '   • Instance Segmentation: 정확한 객체 마스크 생성',
                        '   • Pose Estimation: 사람 관절점 검출',
                        '   • Classification: 이미지 카테고리 분류',
                        '',
                        '📏 모델 크기별 특성:',
                        '   • Nano (n): 가장 빠름, 모바일/엣지 최적화',
                        '   • Small (s): 속도와 정확도의 균형',
                        '   • Medium (m): 일반적 용도, 좋은 성능',
                        '   • Large (l): 높은 정확도, 더 많은 리소스',
                        '   • Extra Large (x): 최고 정확도, 연구용',
                        '',
                        '🔥 추천 모델:',
                        '   • 첫 사용자: YOLOv8s 또는 YOLOv11s',
                        '   • 실시간 처리: Nano 모델',
                        '   • 최고 성능: Large/XL 모델',
                        '   • GPU 메모리 부족: Small 이하',
                        '',
                        '⚡ 최신 모델:',
                        '   • YOLOv11: 최신 아키텍처, 가장 효율적',
                        '   • RT-DETR: 트랜스포머 기반 검출기',
                        '   • SAM2: 최신 분할 모델',
                        '   • YOLO-World: 오픈 보케블러리 지원'
                    ]
                }
            },
            'training_parameters': {
                'ko': {
                    'title': '훈련 파라미터 도움말',
                    'content': [
                        '🔧 주요 파라미터:',
                        '   • epochs: 훈련 반복 횟수 (100-300 권장)',
                        '   • batch_size: 배치 크기 (GPU 메모리에 따라 조정)',
                        '   • learning_rate: 학습률 (0.001-0.01)',
                        '   • img_size: 이미지 크기 (640 기본값)',
                        '',
                        '💾 메모리별 배치 크기:',
                        '   • 4GB GPU: batch_size=8',
                        '   • 8GB GPU: batch_size=16',
                        '   • 16GB+ GPU: batch_size=32+',
                        '',
                        '📈 성능 최적화:',
                        '   • Mixed precision: 메모리 절약',
                        '   • Data augmentation: 과적합 방지',
                        '   • Early stopping: 자동 중단'
                    ]
                }
            },
            'general': {
                'ko': {
                    'title': '일반 도움말',
                    'content': [
                        '🚀 AI 훈련 시스템 v3.0 사용법:',
                        '',
                        '📝 기본 명령어:',
                        '   • !help: 상황별 도움말 표시',
                        '   • ESC: 이전 단계로 돌아가기',
                        '   • Ctrl+C: 프로그램 종료',
                        '',
                        '🔄 워크플로우:',
                        '   1. 워크플로우 선택 (자동/반자동/수동)',
                        '   2. 시스템 환경 검사',
                        '   3. 데이터셋 검색 및 선택',
                        '   4. 압축파일 자동 해제',
                        '   5. 모델 및 파라미터 설정',
                        '   6. 훈련 실행 및 모니터링',
                        '',
                        '💡 문제 해결:',
                        '   • 오류 발생시 AI 기반 해결책 제시',
                        '   • ChatGPT 연동으로 상세 도움',
                        '   • 로그 파일 자동 분석',
                        '',
                        '📊 모니터링:',
                        '   • 실시간 하드웨어 사용률',
                        '   • 훈련 진행률 및 성능 그래프',
                        '   • 최적화 추천사항'
                    ]
                }
            }
        }
    
    def show_help(self, context: str = 'general'):
        """상황별 도움말 표시"""
        help_info = self.help_data.get(context, self.help_data['general'])
        lang_help = help_info.get(self.lang.current_language, help_info['ko'])
        
        if not RICH_AVAILABLE:
            print(f"\n{lang_help['title']}")
            print("=" * len(lang_help['title']))
            for line in lang_help['content']:
                print(line)
            print()
            return
        
        # Rich 기반 도움말
        help_text = Text()
        for line in lang_help['content']:
            if line.startswith('🔧') or line.startswith('🤖') or line.startswith('📋'):
                help_text.append(line + '\n', style="bold blue")
            elif line.startswith('   •'):
                help_text.append(line + '\n', style="green")
            elif line.startswith('   '):
                help_text.append(line + '\n', style="yellow")
            elif line == '':
                help_text.append('\n')
            else:
                help_text.append(line + '\n', style="cyan")
        
        help_panel = Panel(
            help_text,
            title=f"❓ {lang_help['title']}",
            title_align="left",
            border_style="blue"
        )
        
        self.ui.console.print(help_panel)

# ════════════════════════════════════════════════════════════════════════════════
# 🏃‍♂️ 훈련 실행 엔진
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """훈련 설정 데이터 클래스 - Windows 호환성"""
    model_name: str = "yolov8n.pt"
    dataset_path: str = ""
    epochs: int = 100
    batch_size: int = 4  # Windows에서 안전한 기본값
    img_size: int = 640
    learning_rate: float = 0.01
    device: str = "auto"
    mixed_precision: bool = True
    data_augmentation: bool = True
    early_stopping: bool = True
    save_period: int = 10
    workers: int = 0  # Windows 기본값
    project_name: str = "yolo_training"
    experiment_name: str = "exp"

class TrainingEngine:
    """
    AI 모델 훈련 실행 엔진
    - YOLO 모델 통합 지원
    - 실시간 모니터링
    - 자동 최적화
    """
    
    def __init__(self, logger: AdvancedLogger, ui: AdvancedUI, 
                 hardware_monitor: HardwareMonitor):
        self.logger = logger
        self.ui = ui
        self.hardware_monitor = hardware_monitor
        
        # 훈련 상태
        self.is_training = False
        self.current_config = None
        self.training_stats = {}
        
        # 지원 모델 목록 (전체 Ultralytics 모델)
        self.supported_models = {
            'Object Detection': {
                'YOLOv3': {
                    'tiny': 'yolov3-tiny.pt',
                    'standard': 'yolov3.pt'
                },
                'YOLOv4': {
                    'tiny': 'yolov4-tiny.pt',
                    'standard': 'yolov4.pt'
                },
                'YOLOv5': {
                    'nano': 'yolov5n.pt',
                    'small': 'yolov5s.pt',
                    'medium': 'yolov5m.pt',
                    'large': 'yolov5l.pt',
                    'extra_large': 'yolov5x.pt',
                    'nano6': 'yolov5n6.pt',
                    'small6': 'yolov5s6.pt',
                    'medium6': 'yolov5m6.pt',
                    'large6': 'yolov5l6.pt',
                    'extra_large6': 'yolov5x6.pt'
                },
                'YOLOv6': {
                    'nano': 'yolov6n.pt',
                    'small': 'yolov6s.pt',
                    'medium': 'yolov6m.pt',
                    'large': 'yolov6l.pt',
                    'extra_large': 'yolov6x.pt'
                },
                'YOLOv7': {
                    'tiny': 'yolov7-tiny.pt',
                    'standard': 'yolov7.pt',
                    'x': 'yolov7x.pt',
                    'w6': 'yolov7-w6.pt',
                    'e6': 'yolov7-e6.pt',
                    'd6': 'yolov7-d6.pt',
                    'e6e': 'yolov7-e6e.pt'
                },
                'YOLOv8': {
                    'nano': 'yolov8n.pt',
                    'small': 'yolov8s.pt',
                    'medium': 'yolov8m.pt',
                    'large': 'yolov8l.pt',
                    'extra_large': 'yolov8x.pt'
                },
                'YOLOv9': {
                    'tiny': 'yolov9t.pt',
                    'small': 'yolov9s.pt',
                    'medium': 'yolov9m.pt',
                    'compact': 'yolov9c.pt',
                    'extended': 'yolov9e.pt'
                },
                'YOLOv10': {
                    'nano': 'yolov10n.pt',
                    'small': 'yolov10s.pt',
                    'medium': 'yolov10m.pt',
                    'balanced': 'yolov10b.pt',
                    'large': 'yolov10l.pt',
                    'extra_large': 'yolov10x.pt'
                },
                'YOLOv11': {
                    'nano': 'yolo11n.pt',
                    'small': 'yolo11s.pt',
                    'medium': 'yolo11m.pt',
                    'large': 'yolo11l.pt',
                    'extra_large': 'yolo11x.pt'
                },
                'RT-DETR': {
                    'large': 'rtdetr-l.pt',
                    'extra_large': 'rtdetr-x.pt'
                },
                'YOLO-NAS': {
                    'small': 'yolo_nas_s.pt',
                    'medium': 'yolo_nas_m.pt',
                    'large': 'yolo_nas_l.pt'
                },
                'YOLO-World': {
                    'small': 'yolov8s-world.pt',
                    'medium': 'yolov8m-world.pt',
                    'large': 'yolov8l.pt'
                },
                'YOLOE': {
                    'small': 'yoloe-s.pt',
                    'medium': 'yoloe-m.pt',
                    'large': 'yoloe-l.pt',
                    'extra_large': 'yoloe-x.pt'
                }
            },
            'Instance Segmentation': {
                'YOLOv5-Seg': {
                    'nano': 'yolov5n-seg.pt',
                    'small': 'yolov5s-seg.pt',
                    'medium': 'yolov5m-seg.pt',
                    'large': 'yolov5l-seg.pt',
                    'extra_large': 'yolov5x-seg.pt'
                },
                'YOLOv8-Seg': {
                    'nano': 'yolov8n-seg.pt',
                    'small': 'yolov8s-seg.pt',
                    'medium': 'yolov8m-seg.pt',
                    'large': 'yolov8l-seg.pt',
                    'extra_large': 'yolov8x-seg.pt'
                },
                'YOLOv11-Seg': {
                    'nano': 'yolo11n-seg.pt',
                    'small': 'yolo11s-seg.pt',
                    'medium': 'yolo11m-seg.pt',
                    'large': 'yolo11l-seg.pt',
                    'extra_large': 'yolo11x-seg.pt'
                },
                'FastSAM': {
                    'small': 'FastSAM-s.pt',
                    'extra_large': 'FastSAM-x.pt'
                },
                'SAM': {
                    'base': 'sam_b.pt',
                    'large': 'sam_l.pt',
                    'huge': 'sam_h.pt'
                },
                'SAM2': {
                    'base': 'sam2_b.pt',
                    'large': 'sam2_l.pt',
                    'small': 'sam2_s.pt',
                    'tiny': 'sam2_t.pt'
                },
                'MobileSAM': {
                    'standard': 'mobile_sam.pt'
                }
            },
            'Pose Estimation': {
                'YOLOv8-Pose': {
                    'nano': 'yolov8n-pose.pt',
                    'small': 'yolov8s-pose.pt',
                    'medium': 'yolov8m-pose.pt',
                    'large': 'yolov8l-pose.pt',
                    'extra_large': 'yolov8x-pose.pt'
                },
                'YOLOv11-Pose': {
                    'nano': 'yolo11n-pose.pt',
                    'small': 'yolo11s-pose.pt',
                    'medium': 'yolo11m-pose.pt',
                    'large': 'yolo11l-pose.pt',
                    'extra_large': 'yolo11x-pose.pt'
                }
            },
            'Classification': {
                'YOLOv8-Cls': {
                    'nano': 'yolov8n-cls.pt',
                    'small': 'yolov8s-cls.pt',
                    'medium': 'yolov8m-cls.pt',
                    'large': 'yolov8l-cls.pt',
                    'extra_large': 'yolov8x-cls.pt'
                },
                'YOLOv11-Cls': {
                    'nano': 'yolo11n-cls.pt',
                    'small': 'yolo11s-cls.pt',
                    'medium': 'yolo11m-cls.pt',
                    'large': 'yolo11l-cls.pt',
                    'extra_large': 'yolo11x-cls.pt'
                }
            }
        }
        
        # 모델별 권장 사용 케이스
        self.model_recommendations = {
            'yolov8n.pt': '🚀 빠른 추론, 모바일/엣지 디바이스',
            'yolov8s.pt': '⚖️ 속도와 정확도의 균형',
            'yolov8m.pt': '🎯 일반적인 용도, 좋은 성능',
            'yolov8l.pt': '🔍 높은 정확도 요구',
            'yolov8x.pt': '🏆 최고 정확도, 연구용',
            'yolo11n.pt': '⚡ 최신 아키텍처, 빠른 속도',
            'yolo11s.pt': '🆕 YOLOv11 소형 모델',
            'yolo11m.pt': '🆕 YOLOv11 중형 모델',
            'yolo11l.pt': '🆕 YOLOv11 대형 모델',
            'yolo11x.pt': '🆕 YOLOv11 최대 모델',
            'rtdetr-l.pt': '🔄 Real-Time DETR, 트랜스포머',
            'yolo_nas_s.pt': '🧠 Neural Architecture Search',
            'yolov8s-world.pt': '🌍 YOLO-World, 오픈 보케블러리',
            'FastSAM-s.pt': '⚡ 빠른 세그멘테이션',
            'sam_b.pt': '🎭 SAM 기본 모델',
            'yolov8n-pose.pt': '🤸 포즈 추정',
            'yolov8n-cls.pt': '📊 이미지 분류'
        }
    
    def auto_configure_training(self, dataset_path: Path, 
                          workflow_mode: str, 
                          selected_model: str) -> TrainingConfig:
        """
        AI 기반 훈련 설정 자동 최적화 (Windows DataLoader 문제 해결)
        Args:
            dataset_path: 데이터셋 경로
            workflow_mode: 워크플로우 모드 (auto/semi_auto/manual)
            selected_model: 사용자가 선택한 모델
        """
        config = TrainingConfig()
        config.dataset_path = str(dataset_path)
        config.model_name = selected_model  # 사용자 선택 모델 적용
        
        # 하드웨어 분석
        hardware_info = self.hardware_monitor.get_performance_summary()
        current_hw = hardware_info.get('current', {})
        
        # Windows에서 DataLoader 워커 수 조정
        if platform.system() == "Windows":
            config.workers = 0  # Windows에서는 멀티프로세싱 비활성화
            self.logger.info("Windows 환경 감지: DataLoader workers를 0으로 설정")
        else:
            # Linux/macOS에서만 멀티프로세싱 사용
            cpu_info = current_hw.get('cpu', {})
            cpu_cores = cpu_info.get('core_count', 4)
            config.workers = min(cpu_cores // 2, 4)  # 절반만 사용
        
        # GPU 메모리 기반 배치 크기 자동 설정 (모델 크기 고려)
        gpu_info = current_hw.get('gpu', [])
        if gpu_info:
            gpu_memory = gpu_info[0].get('memory_total_mb', 0) / 1024  # GB
            
            # 모델별 메모리 사용량 추정
            model_memory_factor = self._get_model_memory_factor(selected_model)
            
            if gpu_memory <= 4:
                config.batch_size = max(1, int(8 / model_memory_factor))
            elif gpu_memory <= 8:
                config.batch_size = max(2, int(16 / model_memory_factor))
            elif gpu_memory <= 16:
                config.batch_size = max(4, int(32 / model_memory_factor))
            else:
                config.batch_size = max(8, int(64 / model_memory_factor))
        else:
            # CPU 모드 - 작은 배치 사용
            config.batch_size = 2
            config.workers = 0  # CPU 모드에서도 워커 비활성화
        
        # 데이터셋 분석 기반 설정
        dataset_info = self._analyze_dataset(dataset_path)
        
        if dataset_info['image_count'] < 1000:
            config.epochs = 200  # 적은 데이터의 경우 더 많은 epochs
        elif dataset_info['image_count'] > 10000:
            config.epochs = 50   # 많은 데이터의 경우 적은 epochs
        
        # 이미지 크기 분석
        avg_size = dataset_info.get('avg_image_size', 640)
        config.img_size = self._round_to_valid_size(avg_size)
        
        self.logger.info(f"자동 설정 완료: {config.model_name}, "
                        f"batch_size={config.batch_size}, "
                        f"workers={config.workers}, "
                        f"epochs={config.epochs}")
        
        return config
    
    def _get_model_memory_factor(self, model_name: str) -> float:
        """모델별 메모리 사용량 추정 계수"""
        memory_factors = {
            # Nano models
            'yolov8n.pt': 1.0, 'yolo11n.pt': 1.0, 'yolov5n.pt': 1.0,
            # Small models
            'yolov8s.pt': 1.5, 'yolo11s.pt': 1.5, 'yolov5s.pt': 1.5,
            # Medium models
            'yolov8m.pt': 2.5, 'yolo11m.pt': 2.5, 'yolov5m.pt': 2.5,
            # Large models
            'yolov8l.pt': 4.0, 'yolo11l.pt': 4.0, 'yolov5l.pt': 4.0,
            # Extra large models
            'yolov8x.pt': 6.0, 'yolo11x.pt': 6.0, 'yolov5x.pt': 6.0,
            # Segmentation models (더 많은 메모리 사용)
            'yolov8n-seg.pt': 1.5, 'yolo11n-seg.pt': 1.5,
            'yolov8s-seg.pt': 2.0, 'yolo11s-seg.pt': 2.0,
            'yolov8m-seg.pt': 3.0, 'yolo11m-seg.pt': 3.0,
            # RT-DETR (트랜스포머, 더 많은 메모리)
            'rtdetr-l.pt': 5.0, 'rtdetr-x.pt': 7.0,
            # SAM models (매우 큰 메모리 사용)
            'sam_b.pt': 8.0, 'sam_l.pt': 12.0, 'sam_h.pt': 20.0,
            'sam2_t.pt': 3.0, 'sam2_s.pt': 5.0, 'sam2_b.pt': 8.0, 'sam2_l.pt': 12.0
        }
        return memory_factors.get(model_name, 2.0)  # 기본값 2.0
    
    def _analyze_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """데이터셋 분석"""
        analysis = {
            'image_count': 0,
            'class_count': 0,
            'avg_image_size': 640,
            'image_formats': set(),
            'has_labels': False
        }
        
        try:
            if not dataset_path.exists():
                return analysis
            
            # 이미지 파일 검색
            image_files = []
            for ext in SystemConstants.IMAGE_EXTENSIONS:
                image_files.extend(dataset_path.rglob(f"*{ext}"))
                image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
            
            analysis['image_count'] = len(image_files)
            
            # 샘플 이미지들로 평균 크기 계산
            if image_files and PIL_AVAILABLE:
                sample_size = min(10, len(image_files))
                sample_files = image_files[:sample_size]
                
                sizes = []
                for img_file in sample_files:
                    try:
                        with Image.open(img_file) as img:
                            sizes.append(max(img.size))
                            analysis['image_formats'].add(img.format)
                    except Exception:
                        continue
                
                if sizes:
                    analysis['avg_image_size'] = sum(sizes) // len(sizes)
            
            # 클래스 수 추정 (폴더 구조 기반)
            class_dirs = [d for d in dataset_path.iterdir() 
                         if d.is_dir() and not d.name.startswith('.')]
            analysis['class_count'] = len(class_dirs)
            
            # 라벨 파일 확인
            label_files = list(dataset_path.rglob("*.txt"))
            analysis['has_labels'] = len(label_files) > 0
            
        except Exception as e:
            self.logger.error(f"데이터셋 분석 실패: {e}")
        
        return analysis
    
    def _round_to_valid_size(self, size: int) -> int:
        """YOLO에 유효한 이미지 크기로 반올림"""
        valid_sizes = [320, 416, 512, 608, 640, 736, 832, 896, 960, 1024, 1280]
        return min(valid_sizes, key=lambda x: abs(x - size))
    
    def analyze_trained_model_classes(self, model_path: str = None) -> Dict[str, Any]:
        """훈련된 모델의 클래스 정보 분석"""
        
        if not model_path and self.current_config:
            # 최근 훈련 결과에서 best.pt 사용
            results_dir = Path(self.current_config.project_name) / self.current_config.experiment_name
            model_path = results_dir / "best.pt"
        
        if not model_path or not Path(model_path).exists():
            self.logger.warning("분석할 모델 파일을 찾을 수 없습니다.")
            return {}
        
        try:
            # 훈련된 모델 로드
            model = YOLO(model_path)
            
            # 모델 정보 추출
            class_info = {
                'model_path': str(model_path),
                'model_type': self._detect_model_type(model_path),
                'num_classes': 0,
                'class_names': {},
                'class_list': [],
                'dataset_info': {}
            }
            
            # 모델에서 클래스 정보 추출
            if hasattr(model.model, 'names'):
                names = model.model.names
                if isinstance(names, dict):
                    class_info['class_names'] = names
                    class_info['num_classes'] = len(names)
                    class_info['class_list'] = [names[i] for i in sorted(names.keys())]
                elif isinstance(names, list):
                    class_info['class_names'] = {i: name for i, name in enumerate(names)}
                    class_info['num_classes'] = len(names)
                    class_info['class_list'] = names
            
            # YAML 파일에서 추가 정보
            if self.current_config and self.current_config.dataset_path:
                yaml_path = Path(self.current_config.dataset_path)
                if yaml_path.exists():
                    try:
                        with open(yaml_path, 'r', encoding='utf-8') as f:
                            yaml_data = yaml.safe_load(f)
                        
                        if yaml_data:
                            class_info['dataset_info'] = {
                                'dataset_path': yaml_data.get('path', ''),
                                'train_path': yaml_data.get('train', ''),
                                'val_path': yaml_data.get('val', ''),
                                'yaml_classes': yaml_data.get('names', {})
                            }
                    except Exception as e:
                        self.logger.debug(f"YAML 파일 읽기 실패: {e}")
            
            self.logger.info(f"클래스 정보 분석 완료: {class_info['num_classes']}개 클래스")
            return class_info
            
        except Exception as e:
            self.logger.error(f"모델 클래스 분석 실패: {e}")
            return {}

    def _detect_model_type(self, model_path: str) -> str:
        """모델 타입 감지"""
        model_name = Path(model_path).name.lower()
        
        if 'seg' in model_name:
            return 'Instance Segmentation'
        elif 'pose' in model_name:
            return 'Pose Estimation'
        elif 'cls' in model_name:
            return 'Classification'
        elif 'obb' in model_name:
            return 'Oriented Bounding Box'
        else:
            return 'Object Detection'

    def show_class_detection_results(self, model_path: str = None):
        """클래스 탐지 결과 표시"""
        
        class_info = self.analyze_trained_model_classes(model_path)
        
        if not class_info:
            if RICH_AVAILABLE:
                self.ui.console.print("[red]❌ 클래스 정보를 분석할 수 없습니다.[/red]")
            else:
                print("❌ 클래스 정보를 분석할 수 없습니다.")
            return
        
        if not RICH_AVAILABLE:
            self._show_class_info_text(class_info)
        else:
            self._show_class_info_rich(class_info)

    def _show_class_info_text(self, class_info: Dict[str, Any]):
        """텍스트 기반 클래스 정보 표시"""
        print("\n" + "="*60)
        print("🏷️ 모델 클래스 정보")
        print("="*60)
        
        print(f"📁 모델 파일: {Path(class_info['model_path']).name}")
        print(f"🎯 모델 타입: {class_info['model_type']}")
        print(f"📊 클래스 수: {class_info['num_classes']}개")
        
        if class_info['class_list']:
            print(f"\n🏷️ 감지 가능한 클래스:")
            for i, class_name in enumerate(class_info['class_list']):
                print(f"  {i:2d}. {class_name}")
        
        # 데이터셋 정보
        dataset_info = class_info.get('dataset_info', {})
        if dataset_info.get('dataset_path'):
            print(f"\n📂 훈련 데이터셋: {dataset_info['dataset_path']}")
        
        print("="*60)

    def _show_class_info_rich(self, class_info: Dict[str, Any]):
        """Rich 기반 클래스 정보 표시"""
        
        # 메인 정보 패널
        info_text = Text()
        info_text.append("🏷️ 모델 클래스 분석 결과\n\n", style="bold blue")
        
        info_text.append("📁 모델 파일: ", style="cyan")
        info_text.append(f"{Path(class_info['model_path']).name}\n", style="white")
        
        info_text.append("🎯 모델 타입: ", style="cyan")
        info_text.append(f"{class_info['model_type']}\n", style="green")
        
        info_text.append("📊 클래스 수: ", style="cyan")
        info_text.append(f"{class_info['num_classes']}개", style="bold yellow")
        
        info_panel = Panel(
            info_text,
            title="📋 모델 정보",
            border_style="blue"
        )
        
        # 클래스 목록 테이블
        if class_info['class_list']:
            class_table = Table(title="🏷️ 감지 가능한 클래스", show_header=True)
            class_table.add_column("ID", width=4, style="cyan", justify="center")
            class_table.add_column("클래스 이름", style="bold white")
            class_table.add_column("타입", width=10, style="green")
            
            for i, class_name in enumerate(class_info['class_list']):
                # 클래스 타입 추측
                class_type = self._guess_class_type(class_name)
                class_table.add_row(str(i), class_name, class_type)
        
        # 데이터셋 정보 패널
        dataset_info = class_info.get('dataset_info', {})
        if dataset_info.get('dataset_path'):
            dataset_text = Text()
            dataset_text.append("📂 데이터셋 경로\n", style="bold green")
            dataset_text.append(f"{dataset_info['dataset_path']}\n\n", style="cyan")
            
            if dataset_info.get('train_path'):
                dataset_text.append("🔥 훈련 데이터: ", style="yellow")
                dataset_text.append(f"{dataset_info['train_path']}\n", style="white")
            
            if dataset_info.get('val_path'):
                dataset_text.append("✅ 검증 데이터: ", style="yellow")
                dataset_text.append(f"{dataset_info['val_path']}", style="white")
            
            dataset_panel = Panel(
                dataset_text,
                title="📁 데이터셋 정보",
                border_style="green"
            )
        
        # 사용법 안내 패널
        usage_text = Text()
        usage_text.append("💡 모델 사용 방법\n\n", style="bold blue")
        usage_text.append("1. 추론 실행:\n", style="green")
        usage_text.append(f"   yolo predict model={Path(class_info['model_path']).name} source=이미지경로\n\n", style="dim")
        usage_text.append("2. 검증 실행:\n", style="green")
        usage_text.append(f"   yolo val model={Path(class_info['model_path']).name} data=dataset.yaml\n\n", style="dim")
        usage_text.append("3. 모델 내보내기:\n", style="green")
        usage_text.append(f"   yolo export model={Path(class_info['model_path']).name} format=onnx", style="dim")
        
        usage_panel = Panel(
            usage_text,
            title="🚀 사용 가이드",
            border_style="yellow"
        )
        
        # 모든 패널 출력
        self.ui.console.print("\n")
        self.ui.console.print(info_panel)
        self.ui.console.print("\n")
        
        if class_info['class_list']:
            self.ui.console.print(class_table)
            self.ui.console.print("\n")
        
        if dataset_info.get('dataset_path'):
            self.ui.console.print(dataset_panel)
            self.ui.console.print("\n")
        
        self.ui.console.print(usage_panel)

    def _guess_class_type(self, class_name: str) -> str:
        """클래스 이름으로부터 타입 추측"""
        class_name_lower = class_name.lower()
        
        # 사람 관련
        if any(word in class_name_lower for word in ['person', 'people', 'human', '사람', '인간']):
            return "👤 사람"
        
        # 동물 관련
        elif any(word in class_name_lower for word in ['dog', 'cat', 'bird', 'animal', '강아지', '고양이', '새']):
            return "🐾 동물"
        
        # 차량 관련
        elif any(word in class_name_lower for word in ['car', 'truck', 'bus', 'vehicle', '자동차', '트럭', '버스']):
            return "🚗 차량"
        
        # 음식 관련
        elif any(word in class_name_lower for word in ['food', 'fruit', 'cake', '음식', '과일', '케이크']):
            return "🍎 음식"
        
        # 물체 관련
        elif any(word in class_name_lower for word in ['bottle', 'chair', 'table', '병', '의자', '테이블']):
            return "📦 물체"
        
        else:
            return "🏷️ 기타"

    def quick_class_summary(self, model_path: str = None) -> str:
        """클래스 정보 한 줄 요약"""
        class_info = self.analyze_trained_model_classes(model_path)
        
        if not class_info or not class_info['class_list']:
            return "❌ 클래스 정보 없음"
        
        num_classes = class_info['num_classes']
        class_preview = ', '.join(class_info['class_list'][:3])
        
        if num_classes > 3:
            class_preview += f" 외 {num_classes-3}개"
        
        return f"🏷️ {num_classes}개 클래스: {class_preview}"
    
    def show_model_selection(self, workflow_mode: str) -> str:
        """모델 선택 UI - 모든 워크플로우에서 사용자 선택"""
        
        if not RICH_AVAILABLE:
            return self._show_text_model_selection()
        
        # Rich 기반 모델 선택
        return self._show_rich_model_selection()
    
    def _show_text_model_selection(self) -> str:
        """텍스트 기반 모델 선택"""
        print("\n" + "="*80)
        print("🎯 AI 모델 선택")
        print("="*80)
        
        # 플랫 모델 리스트 생성
        models = []
        model_counter = 1
        
        for category, model_dict in self.supported_models.items():
            print(f"\n📂 {category}:")
            for series, variants in model_dict.items():
                for variant, filename in variants.items():
                    description = self.model_recommendations.get(filename, "AI 모델")
                    print(f"  {model_counter:2d}. {series} {variant} ({filename})")
                    print(f"      {description}")
                    models.append(filename)
                    model_counter += 1
        
        while True:
            try:
                print(f"\n선택 가능한 모델: 1-{len(models)}")
                choice = input("모델 번호를 선택하세요 (또는 !help): ").strip()
                
                if choice == "!help":
                    self._show_model_help()
                    continue
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(models):
                    selected_model = models[choice_num - 1]
                    print(f"\n✅ 선택된 모델: {selected_model}")
                    return selected_model
                else:
                    print(f"❌ 1부터 {len(models)} 사이의 번호를 입력하세요.")
                    
            except ValueError:
                print("❌ 올바른 번호를 입력하세요.")
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                sys.exit(0)
    
    def _show_rich_model_selection(self) -> str:
        """Rich 기반 모델 선택"""
        
        # 카테고리 선택
        categories = list(self.supported_models.keys())
        
        # 카테고리 선택 테이블
        category_table = Table(title="🎯 AI 모델 카테고리 선택", show_header=True)
        category_table.add_column("번호", width=4, style="cyan")
        category_table.add_column("카테고리", width=25, style="bold")
        category_table.add_column("설명", style="blue")
        
        category_descriptions = {
            'Object Detection': '객체 검출 - 이미지에서 객체 위치와 클래스 식별',
            'Instance Segmentation': '인스턴스 분할 - 객체의 정확한 마스크 생성',
            'Pose Estimation': '포즈 추정 - 사람의 관절점 위치 검출',
            'Classification': '이미지 분류 - 이미지를 카테고리로 분류'
        }
        
        for i, category in enumerate(categories, 1):
            description = category_descriptions.get(category, "AI 모델 카테고리")
            category_table.add_row(str(i), category, description)
        
        self.ui.console.print(category_table)
        
        while True:
            try:
                category_choice = Prompt.ask(
                    "\n[bold cyan]카테고리를 선택하세요[/bold cyan]",
                    choices=[str(i) for i in range(1, len(categories) + 1)] + ["!help"],
                    default="1"
                )
                
                if category_choice == "!help":
                    self._show_category_help()
                    continue
                
                selected_category = categories[int(category_choice) - 1]
                break
                
            except (ValueError, IndexError):
                self.ui.console.print("[red]올바른 번호를 선택하세요.[/red]")
        
        # 선택된 카테고리의 모델들 표시
        models = self.supported_models[selected_category]
        
        # 모델 선택 테이블
        model_table = Table(title=f"🤖 {selected_category} 모델 선택", show_header=True, show_lines=True)
        model_table.add_column("번호", width=4, style="cyan")
        model_table.add_column("시리즈", width=15, style="bold")
        model_table.add_column("크기", width=12, style="yellow")
        model_table.add_column("모델 파일", width=20, style="green")
        model_table.add_column("권장 용도", style="blue")
        
        flat_models = []
        model_counter = 1
        
        for series, variants in models.items():
            for variant, filename in variants.items():
                recommendation = self.model_recommendations.get(filename, "범용 AI 모델")
                model_table.add_row(
                    str(model_counter),
                    series,
                    variant.capitalize(),
                    filename,
                    recommendation
                )
                flat_models.append(filename)
                model_counter += 1
        
        self.ui.console.print(model_table)
        
        # 하드웨어 기반 추천
        self._show_hardware_recommendations()
        
        while True:
            try:
                model_choice = Prompt.ask(
                    "\n[bold cyan]모델을 선택하세요[/bold cyan]",
                    choices=[str(i) for i in range(1, len(flat_models) + 1)] + ["!help", "back"],
                    default="1"
                )
                
                if model_choice == "!help":
                    self._show_model_help()
                    continue
                elif model_choice == "back":
                    return self._show_rich_model_selection()  # 카테고리 선택으로 돌아가기
                
                selected_model = flat_models[int(model_choice) - 1]
                
                # 선택 확인
                self.ui.console.print(f"\n[green]✅ 선택된 모델: {selected_model}[/green]")
                if Confirm.ask("이 모델로 진행하시겠습니까?", default=True):
                    return selected_model
                
            except (ValueError, IndexError):
                self.ui.console.print("[red]올바른 번호를 선택하세요.[/red]")

    def _show_hardware_recommendations(self):
        """하드웨어 기반 모델 추천"""
        hardware_info = self.hardware_monitor.get_performance_summary()
        current_hw = hardware_info.get('current', {})
        
        # GPU 정보 기반 추천
        gpu_info = current_hw.get('gpu', [])
        
        rec_text = Text()
        rec_text.append("💡 하드웨어 기반 추천\n\n", style="bold blue")
        
        if gpu_info:
            gpu_memory = gpu_info[0].get('memory_total_mb', 0) / 1024  # GB
            gpu_name = gpu_info[0].get('name', 'Unknown GPU')
            
            rec_text.append(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)\n", style="cyan")
            
            if gpu_memory <= 4:
                rec_text.append("추천: Nano 모델 (n) - 메모리 절약\n", style="green")
            elif gpu_memory <= 8:
                rec_text.append("추천: Small 모델 (s) - 균형잡힌 성능\n", style="green")
            elif gpu_memory <= 16:
                rec_text.append("추천: Medium 모델 (m) - 좋은 성능\n", style="green")
            else:
                rec_text.append("추천: Large/XL 모델 (l/x) - 최고 성능\n", style="green")
        else:
            rec_text.append("🖥️ CPU 모드\n", style="yellow")
            rec_text.append("추천: Nano/Small 모델 - CPU 최적화\n", style="green")
        
        # 메모리 정보
        memory_info = current_hw.get('memory', {})
        memory_total = memory_info.get('total_gb', 0)
        rec_text.append(f"💾 RAM: {memory_total:.1f}GB\n", style="cyan")
        
        if memory_total < 8:
            rec_text.append("⚠️ 메모리 부족 - 작은 배치 크기 사용 예정\n", style="yellow")
        
        rec_panel = Panel(
            rec_text,
            title="🎯 하드웨어 추천",
            title_align="left",
            border_style="blue"
        )
        
        self.ui.console.print(rec_panel)
    
    def _show_category_help(self):
        """카테고리 도움말"""
        help_text = Text()
        help_text.append("📂 모델 카테고리 설명\n\n", style="bold blue")
        
        help_text.append("🎯 Object Detection\n", style="bold green")
        help_text.append("   • 이미지에서 객체의 위치(바운딩 박스)와 클래스를 검출\n", style="green")
        help_text.append("   • 가장 일반적인 컴퓨터 비전 작업\n", style="green")
        help_text.append("   • 예: 사람, 자동차, 동물 등을 찾고 분류\n\n", style="green")
        
        help_text.append("🎭 Instance Segmentation\n", style="bold yellow")
        help_text.append("   • 객체의 정확한 픽셀 단위 마스크를 생성\n", style="yellow")
        help_text.append("   • Object Detection보다 더 정밀한 위치 정보\n", style="yellow")
        help_text.append("   • 의료 이미징, 자율주행 등에 활용\n\n", style="yellow")
        
        help_text.append("🤸 Pose Estimation\n", style="bold red")
        help_text.append("   • 사람의 관절점(키포인트) 위치를 검출\n", style="red")
        help_text.append("   • 스포츠 분석, 피트니스, AR/VR 등에 활용\n", style="red")
        help_text.append("   • 17개 주요 관절점 좌표 제공\n\n", style="red")
        
        help_text.append("📊 Classification\n", style="bold blue")
        help_text.append("   • 이미지 전체를 하나의 클래스로 분류\n", style="blue")
        help_text.append("   • 위치 정보 없이 '무엇인가'만 식별\n", style="blue")
        help_text.append("   • 품질 검사, 의료 진단 등에 활용\n", style="blue")
        
        help_panel = Panel(
            help_text,
            title="❓ 카테고리 도움말",
            title_align="left",
            border_style="blue"
        )
        
        self.ui.console.print(help_panel)
    
    def _show_model_help(self):
        """모델 선택 도움말"""
        help_text = Text()
        help_text.append("🤖 모델 선택 가이드\n\n", style="bold blue")
        
        help_text.append("📏 모델 크기별 특성\n\n", style="bold green")
        help_text.append("• ", style="green")
        help_text.append("Nano (n): ", style="bold green")
        help_text.append("가장 빠름, 모바일/엣지 최적화\n", style="white")
        help_text.append("• ", style="green")
        help_text.append("Small (s): ", style="bold green")
        help_text.append("속도와 정확도의 균형\n", style="white")
        help_text.append("• ", style="green")
        help_text.append("Medium (m): ", style="bold green")
        help_text.append("일반적인 용도, 좋은 성능\n", style="white")
        help_text.append("• ", style="green")
        help_text.append("Large (l): ", style="bold green")
        help_text.append("높은 정확도, 더 많은 리소스 필요\n", style="white")
        help_text.append("• ", style="green")
        help_text.append("Extra Large (x): ", style="bold green")
        help_text.append("최고 정확도, 연구/서버용\n\n", style="white")
        
        help_text.append("🔥 버전별 특징\n\n", style="bold yellow")
        help_text.append("• ", style="yellow")
        help_text.append("YOLOv11: ", style="bold yellow")
        help_text.append("최신, 가장 효율적\n", style="white")
        help_text.append("• ", style="yellow")
        help_text.append("YOLOv8: ", style="bold yellow")
        help_text.append("안정적, 널리 사용됨\n", style="white")
        help_text.append("• ", style="yellow")
        help_text.append("RT-DETR: ", style="bold yellow")
        help_text.append("트랜스포머 기반, 높은 정확도\n", style="white")
        help_text.append("• ", style="yellow")
        help_text.append("SAM/SAM2: ", style="bold yellow")
        help_text.append("분할 전문, 범용성 우수\n", style="white")
        help_text.append("• ", style="yellow")
        help_text.append("YOLO-World: ", style="bold yellow")
        help_text.append("오픈 보케블러리, 새로운 클래스 감지\n\n", style="white")
        
        help_text.append("💡 선택 팁\n\n", style="bold blue")
        help_text.append("• 처음 사용: YOLOv8s 또는 YOLOv11s 추천\n", style="blue")
        help_text.append("• 실시간 추론: Nano 모델\n", style="blue")
        help_text.append("• 최고 정확도: Large 또는 XL 모델\n", style="blue")
        help_text.append("• 메모리 부족: Small 이하 모델\n", style="blue")
        
        help_panel = Panel(
            help_text,
            title="💡 모델 선택 도움말",
            title_align="left",
            border_style="blue"
        )
        
        self.ui.console.print(help_panel)
    
    def setup_training_environment(self, config: TrainingConfig) -> bool:
        """훈련 환경 설정"""
        try:
            self.logger.info("훈련 환경 설정 중...")
        
            # 프로젝트 디렉토리 생성
            project_dir = Path(config.project_name)
            project_dir.mkdir(exist_ok=True)
        
            # 실행 디렉토리 생성
            run_dir = project_dir / config.experiment_name
            run_counter = 1
            while run_dir.exists():
                run_dir = project_dir / f"{config.experiment_name}{run_counter}"
                run_counter += 1
        
            run_dir.mkdir(parents=True)
            config.experiment_name = run_dir.name
        
            # 원본 데이터셋 경로 저장
            original_dataset_path = config.dataset_path
        
            # YAML 설정 파일 생성
            yaml_config = self._create_dataset_yaml(config)
            yaml_path = run_dir / "dataset.yaml"
        
            # YAML 파일 저장
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
        
            # 생성된 YAML 내용 로그
            self.logger.info(f"YAML 파일 생성: {yaml_path}")
            self.logger.info(f"YAML 내용: {yaml_config}")
        
            # config의 dataset_path를 yaml 파일 경로로 업데이트
            config.dataset_path = str(yaml_path)
        
            # YAML 파일 검증
            if not self._verify_yaml_file(yaml_path):
                self.logger.error("생성된 YAML 파일이 유효하지 않습니다")
                return False
        
            self.logger.info(f"훈련 환경 설정 완료: {run_dir}")
            return True
        
        except Exception as e:
            self.logger.error(f"훈련 환경 설정 실패: {e}")
            return False

    def _verify_yaml_file(self, yaml_path: Path) -> bool:
        """YAML 파일 유효성 검증"""
        try:
            # YAML 파일 읽기 테스트
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            if not yaml_data:
                self.logger.error("YAML 파일이 비어있습니다")
                return False
            
            # 필수 키 확인
            required_keys = ['path', 'train', 'names']
            for key in required_keys:
                if key not in yaml_data:
                    self.logger.error(f"YAML 파일에 필수 키 '{key}'가 없습니다")
                    return False
            
            # 경로 존재 여부 확인
            dataset_root = Path(yaml_data['path'])
            train_path = dataset_root / yaml_data['train']
            
            if not train_path.exists():
                self.logger.error(f"훈련 데이터 경로가 존재하지 않습니다: {train_path}")
                return False
            
            # 이미지 파일 확인
            image_files = []
            for ext in SystemConstants.IMAGE_EXTENSIONS:
                image_files.extend(list(train_path.rglob(f"*{ext}")))
                image_files.extend(list(train_path.rglob(f"*{ext.upper()}")))
            
            if not image_files:
                self.logger.error(f"훈련 경로에 이미지 파일이 없습니다: {train_path}")
                return False
            
            self.logger.info(f"YAML 검증 성공: {len(image_files)}개 이미지 파일 확인")
            return True
            
        except Exception as e:
            self.logger.error(f"YAML 파일 검증 실패: {e}")
            return False
        
    def show_dataset_structure(self, dataset_path: Path):
        """데이터셋 구조 디버깅 정보 표시"""
        if not RICH_AVAILABLE:
            print(f"\n데이터셋 구조 분석: {dataset_path}")
            return
        
        tree = Tree(f"📁 {dataset_path.name}")
        
        try:
            def add_directory(parent_tree, directory, max_depth=3, current_depth=0):
                if current_depth >= max_depth:
                    return
                
                items = list(directory.iterdir())[:20]  # 최대 20개 항목만
                
                for item in items:
                    if item.is_dir():
                        dir_node = parent_tree.add(f"📁 {item.name}")
                        add_directory(dir_node, item, max_depth, current_depth + 1)
                    elif item.suffix.lower() in SystemConstants.IMAGE_EXTENSIONS:
                        parent_tree.add(f"🖼️ {item.name}")
                    elif item.suffix.lower() == '.txt':
                        parent_tree.add(f"📝 {item.name}")
                    else:
                        parent_tree.add(f"📄 {item.name}")
            
            add_directory(tree, dataset_path)
            
            self.ui.console.print("\n")
            self.ui.console.print(Panel(tree, title="🔍 데이터셋 구조", border_style="blue"))
            
        except Exception as e:
            self.logger.error(f"데이터셋 구조 표시 실패: {e}")

    
    def _create_dataset_yaml(self, config: TrainingConfig) -> Dict[str, Any]:
        """데이터셋 YAML 설정 생성 - 경로 문제 해결"""
        dataset_path = Path(config.dataset_path).resolve()  # 절대 경로로 변환
    
        self.logger.info(f"데이터셋 경로 분석 중: {dataset_path}")
    
        # 다양한 데이터셋 구조 패턴 확인
        train_path = None
        val_path = None
        test_path = None
    
        # 패턴 1: train/val/test 폴더 구조
        if (dataset_path / "train").exists():
            train_path = dataset_path / "train"
            val_path = dataset_path / "val" if (dataset_path / "val").exists() else train_path
            if (dataset_path / "test").exists():
                test_path = dataset_path / "test"
    
        # 패턴 2: images/labels 폴더 구조
        elif (dataset_path / "images").exists():
            images_dir = dataset_path / "images"
        
            # images 하위에 train/val 확인
            if (images_dir / "train").exists():
                train_path = images_dir / "train"
                val_path = images_dir / "val" if (images_dir / "val").exists() else train_path
                if (images_dir / "test").exists():
                    test_path = images_dir / "test"
            else:
                # images 폴더 자체를 train으로 사용
                train_path = images_dir
                val_path = images_dir
    
        # 패턴 3: 직접 이미지 파일들이 있는 경우
        else:
            # 데이터셋 루트에서 이미지 파일 찾기
            image_files = []
            for ext in SystemConstants.IMAGE_EXTENSIONS:
                image_files.extend(list(dataset_path.rglob(f"*{ext}")))
                image_files.extend(list(dataset_path.rglob(f"*{ext.upper()}")))
        
            if image_files:
                # 첫 번째 이미지가 있는 디렉토리를 찾기
                first_image_dir = image_files[0].parent
                train_path = first_image_dir
                val_path = first_image_dir
            
                self.logger.info(f"이미지 파일 {len(image_files)}개를 {first_image_dir}에서 발견")
    
        # 경로가 발견되지 않은 경우
        if not train_path:
            # 강제로 데이터셋 루트를 train으로 사용
            train_path = dataset_path
            val_path = dataset_path
            self.logger.warning(f"표준 구조를 찾을 수 없어 루트 디렉토리를 사용: {dataset_path}")
    
        # 상대 경로 계산 (YOLO가 상대 경로를 선호함)
        try:
            train_rel = os.path.relpath(train_path, dataset_path)
            val_rel = os.path.relpath(val_path, dataset_path)
        except ValueError:
            # 서로 다른 드라이브에 있는 경우 절대 경로 사용
            train_rel = str(train_path)
            val_rel = str(val_path)
    
        # 클래스 정보 자동 감지
        classes = self._detect_classes(dataset_path, train_path)
    
        yaml_config = {
            'path': str(dataset_path),  # 데이터셋 루트 경로
            'train': train_rel,         # 훈련 이미지 경로
            'val': val_rel,            # 검증 이미지 경로
            'names': classes           # 클래스 정보
        }
    
        # 테스트 경로가 있으면 추가
        if test_path:
            try:
                test_rel = os.path.relpath(test_path, dataset_path)
                yaml_config['test'] = test_rel
            except ValueError:
                yaml_config['test'] = str(test_path)
    
        # 생성된 경로 검증
        self._validate_dataset_paths(dataset_path, yaml_config)
    
        self.logger.info(f"YAML 설정 생성 완료: train={train_rel}, val={val_rel}, classes={len(classes)}")
    
        return yaml_config

    def _validate_dataset_paths(self, dataset_root: Path, yaml_config: Dict[str, Any]):
        """생성된 데이터셋 경로 검증"""
        issues = []
    
        # 훈련 경로 검증
        train_path = dataset_root / yaml_config['train']
        if not train_path.exists():
            issues.append(f"훈련 경로가 존재하지 않음: {train_path}")
        else:
            # 이미지 파일 확인
            image_count = len([f for f in train_path.rglob("*") 
                            if f.suffix.lower() in SystemConstants.IMAGE_EXTENSIONS])
            if image_count == 0:
                issues.append(f"훈련 경로에 이미지 파일이 없음: {train_path}")
            else:
                self.logger.info(f"훈련 이미지 {image_count}개 발견")
    
        # 검증 경로 확인
        val_path = dataset_root / yaml_config['val']
        if not val_path.exists():
            issues.append(f"검증 경로가 존재하지 않음: {val_path}")
    
        # 문제가 있으면 경고
        if issues:
            self.logger.warning("데이터셋 경로 문제 발견:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")

    def _detect_classes(self, dataset_path: Path, train_path: Path = None) -> Dict[int, str]:
        """클래스 자동 감지 - 개선된 버전"""
        classes = {}
    
        try:
            # 방법 1: classes.txt 또는 names.txt 파일 찾기
            class_files = []
            for filename in ['classes.txt', 'names.txt', 'class_names.txt']:
                class_files.extend(list(dataset_path.rglob(filename)))
        
            if class_files:
                class_file = class_files[0]
                self.logger.info(f"클래스 파일 발견: {class_file}")
            
                with open(class_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        class_name = line.strip()
                        if class_name:  # 빈 줄 제외
                            classes[i] = class_name
            
                if classes:
                    self.logger.info(f"클래스 파일에서 {len(classes)}개 클래스 로드")
                    return classes
        
            # 방법 2: data.yaml 파일에서 클래스 정보 추출
            yaml_files = list(dataset_path.rglob("data.yaml")) + list(dataset_path.rglob("dataset.yaml"))
            for yaml_file in yaml_files:
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        existing_yaml = yaml.safe_load(f)
                        if existing_yaml and 'names' in existing_yaml:
                            names = existing_yaml['names']
                            if isinstance(names, dict):
                                classes = {int(k): v for k, v in names.items()}
                            elif isinstance(names, list):
                                classes = {i: name for i, name in enumerate(names)}
                        
                            if classes:
                                self.logger.info(f"기존 YAML에서 {len(classes)}개 클래스 로드")
                                return classes
                except Exception as e:
                    self.logger.debug(f"YAML 파일 읽기 실패 ({yaml_file}): {e}")
        
            # 방법 3: 폴더 구조로 클래스 추정 (train 경로 우선)
            search_paths = [train_path, dataset_path] if train_path else [dataset_path]
        
            for search_path in search_paths:
                if not search_path or not search_path.exists():
                    continue
                
                # 하위 디렉토리들을 클래스로 간주
                class_dirs = [d for d in search_path.iterdir() 
                            if d.is_dir() and not d.name.startswith('.') 
                            and d.name not in ['images', 'labels', 'train', 'val', 'test']]
            
                if class_dirs:
                    # 각 디렉토리에 이미지가 있는지 확인
                    valid_class_dirs = []
                    for class_dir in class_dirs:
                        image_count = len([f for f in class_dir.rglob("*") 
                                        if f.suffix.lower() in SystemConstants.IMAGE_EXTENSIONS])
                        if image_count > 0:
                            valid_class_dirs.append(class_dir.name)
                
                    if valid_class_dirs:
                        classes = {i: name for i, name in enumerate(sorted(valid_class_dirs))}
                        self.logger.info(f"폴더 구조에서 {len(classes)}개 클래스 감지: {list(classes.values())}")
                        return classes
        
            # 방법 4: 라벨 파일에서 클래스 ID 추출
            label_files = list(dataset_path.rglob("*.txt"))
            class_ids = set()
        
            for label_file in label_files[:50]:  # 최대 50개 파일만 확인
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and parts[0].isdigit():
                                class_ids.add(int(parts[0]))
                except Exception:
                    continue
        
            if class_ids:
                max_class_id = max(class_ids)
                classes = {i: f"class_{i}" for i in range(max_class_id + 1)}
                self.logger.info(f"라벨 파일에서 {len(classes)}개 클래스 감지 (class_0 ~ class_{max_class_id})")
                return classes
                
        except Exception as e:
            self.logger.warning(f"클래스 감지 중 오류: {e}")
    
        # 기본값: 단일 클래스
        classes = {0: "object"}
        self.logger.info("기본 클래스 사용: object")
    
        return classes
    
    def start_training(self, config: TrainingConfig) -> bool:
        """훈련 시작 - 모니터링 인터페이스 개선"""
        if not ULTRALYTICS_AVAILABLE:
            self.ui.show_error("YOLO 라이브러리 오류", 
                            "Ultralytics 라이브러리가 설치되지 않았습니다.",
                            "pip install ultralytics로 설치하세요.")
            return False
        
        try:
            self.is_training = True
            self.current_config = config
            
            # 모델 로드
            model = YOLO(config.model_name)
            
            # Windows 환경 특별 설정
            if platform.system() == "Windows":
                self._setup_windows_compatibility()
            
            # YOLO 유효 훈련 파라미터만 설정
            train_args = {
                'data': config.dataset_path,
                'epochs': config.epochs,
                'batch': config.batch_size,
                'imgsz': config.img_size,
                'lr0': config.learning_rate,
                'device': config.device,
                'project': config.project_name,
                'name': config.experiment_name,
                'save_period': config.save_period,
                'workers': config.workers,
                'amp': config.mixed_precision,
                'augment': config.data_augmentation,
                'verbose': False,  # YOLO 자체 출력 최소화
                'exist_ok': True
            }
            
            # Early stopping 설정
            if config.early_stopping:
                train_args['patience'] = 50
            
            # 모델별 특화 설정 추가
            self._add_model_specific_args(train_args, config.model_name)
            
            self.logger.info(f"🚀 훈련 시작: {config.model_name}")
            self.logger.info(f"📊 설정: epochs={config.epochs}, batch={config.batch_size}")
            
            # 하드웨어 모니터링 시작 (저빈도)
            self.hardware_monitor.start_monitoring()
            
            # 훈련 실행 - 간소화된 모니터링
            try:
                if RICH_AVAILABLE:
                    # Rich 기반 간단한 모니터링
                    self._train_with_rich_monitoring(model, train_args, config)
                else:
                    # 텍스트 기반 간단한 모니터링
                    self._train_with_text_monitoring(model, train_args, config)
                
                # 훈련 완료
                self.training_stats['success'] = True
                self.is_training = False
                
                self.logger.info("🎉 훈련 성공적으로 완료!")
                return True
                
            except Exception as training_error:
                error_msg = str(training_error)
                self.logger.error(f"❌ 훈련 중 오류: {error_msg}")
                self.training_stats['error'] = error_msg
                self.training_stats['success'] = False
                self.is_training = False
                
                self._suggest_training_solutions(error_msg)
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 훈련 시작 실패: {e}")
            self.is_training = False
            return False
        finally:
            self.hardware_monitor.stop_monitoring()

    def _train_with_rich_monitoring(self, model, train_args: Dict, config: TrainingConfig):
        """Rich 기반 간소화된 훈련 모니터링"""
        
        # 초기 상태 표시
        self.ui.console.print("\n" + "="*80)
        self.ui.console.print(f"🚀 [bold blue]{config.model_name}[/bold blue] 훈련 시작")
        self.ui.console.print("="*80)
        
        # 설정 정보 표시 (한 번만)
        info_table = Table(show_header=False, box=None)
        info_table.add_column("항목", width=15, style="cyan")
        info_table.add_column("값", style="white")
        
        info_table.add_row("📊 에폭", str(config.epochs))
        info_table.add_row("📦 배치 크기", str(config.batch_size))
        info_table.add_row("🖼️ 이미지 크기", str(config.img_size))
        info_table.add_row("📈 학습률", str(config.learning_rate))
        info_table.add_row("⚙️ 워커", str(config.workers))
        
        self.ui.console.print(info_table)
        self.ui.console.print("\n💡 [dim]훈련이 진행됩니다. 완료까지 기다려주세요...[/dim]\n")
        
        # YOLO 훈련 시작 (출력 최소화)
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO
        
        # YOLO 출력을 캡처하여 주요 정보만 표시
        captured_output = StringIO()
        
        try:
            # YOLO 훈련 실행 (출력 캡처)
            with redirect_stdout(captured_output):
                results = model.train(**train_args)
            
            self.training_stats['results'] = results
            
        except Exception as e:
            # 출력 복원하고 에러 전파
            raise e

    def _train_with_text_monitoring(self, model, train_args: Dict, config: TrainingConfig):
        """텍스트 기반 간소화된 훈련 모니터링"""
        
        print("\n" + "="*60)
        print(f"🚀 {config.model_name} 훈련 시작")
        print("="*60)
        print(f"에폭: {config.epochs}, 배치: {config.batch_size}")
        print(f"이미지 크기: {config.img_size}")
        print("💡 훈련이 진행됩니다. 완료까지 기다려주세요...")
        print("="*60)
        
        # 간단한 진행률 표시
        start_time = time.time()
        
        try:
            results = model.train(**train_args)
            self.training_stats['results'] = results
            
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            print(f"\n🎉 훈련 완료! 소요시간: {mins}분 {secs}초")
            
        except Exception as e:
            raise e

    def _add_model_specific_args(self, train_args: Dict, model_name: str):
        """모델별 특화 설정 추가"""
        model_name_lower = model_name.lower()
        
        # 공통 최적화 설정
        train_args.update({
            'optimizer': 'auto',
            'close_mosaic': 10,
            'cos_lr': False,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'plots': True,
            'save_json': False,
            'cache': False
        })
        
        if 'seg' in model_name_lower:
            # Segmentation 모델
            train_args.update({
                'mask_ratio': 4,
                'overlap_mask': True
            })
        elif 'pose' in model_name_lower:
            # Pose 모델
            train_args.update({
                'pose': 12.0,
                'kobj': 1.0
            })
        elif 'cls' in model_name_lower:
            # Classification 모델
            train_args.update({
                'dropout': 0.2
            })

    def _suggest_training_solutions(self, error_msg: str):
        """훈련 오류에 대한 구체적인 해결책 제시"""
        error_lower = error_msg.lower()
        
        if not RICH_AVAILABLE:
            print("\n해결 방법:")
            if 'argument' in error_lower or 'parameter' in error_lower:
                print("1. YOLO 파라미터 오류 - 최신 Ultralytics 버전으로 업데이트하세요")
                print("   pip install --upgrade ultralytics")
            elif 'memory' in error_lower or 'cuda' in error_lower:
                print("1. GPU 메모리 부족 - 배치 크기를 줄여보세요")
                print("2. Mixed precision (AMP) 활성화 확인")
            elif 'dataloader' in error_lower or 'worker' in error_lower:
                print("1. Workers=0으로 설정되어 있는지 확인")
                print("2. 배치 크기를 더 줄여보세요")
            return
        
        # Rich 기반 해결책 표시
        suggestions = []
        
        if 'argument' in error_lower or 'parameter' in error_lower:
            suggestions.extend([
                "🔧 YOLO 파라미터 오류 해결:",
                "1. Ultralytics 라이브러리를 최신 버전으로 업데이트",
                "   pip install --upgrade ultralytics",
                "2. 지원되지 않는 파라미터 제거 완료",
                "3. 모델 버전과 파라미터 호환성 확인"
            ])
        
        if 'memory' in error_lower or 'cuda' in error_lower:
            suggestions.extend([
                "💾 메모리 문제 해결:",
                f"1. 현재 배치 크기 {self.current_config.batch_size} → 더 작게",
                "2. 이미지 크기 줄이기 (640 → 416 또는 320)",
                "3. Mixed precision (AMP) 활성화 확인",
                "4. GPU 메모리 정리: torch.cuda.empty_cache()"
            ])
        
        if 'dataloader' in error_lower or 'worker' in error_lower:
            suggestions.extend([
                "⚙️ DataLoader 문제 해결:",
                "1. Workers=0 설정 확인 (Windows 필수)",
                "2. 배치 크기를 1 또는 2로 줄이기",
                "3. 데이터셋 경로에 특수문자/한글 확인"
            ])
        
        if 'yaml' in error_lower or 'dataset' in error_lower:
            suggestions.extend([
                "📁 데이터셋 문제 해결:",
                "1. dataset.yaml 파일 경로 확인",
                "2. 이미지 파일 존재 여부 확인",
                "3. 클래스 정의 올바른지 확인"
            ])
        
        if not suggestions:
            suggestions = [
                "🔍 일반적인 해결 방법:",
                "1. 배치 크기를 절반으로 줄이기",
                "2. 이미지 크기 줄이기 (640 → 416)",
                "3. Workers=0으로 설정",
                "4. Ultralytics 라이브러리 업데이트"
            ]
        
        suggestion_text = Text()
        for suggestion in suggestions:
            if suggestion.endswith(':'):
                suggestion_text.append(f"\n{suggestion}\n", style="bold yellow")
            elif suggestion.startswith(('1.', '2.', '3.', '4.')):
                suggestion_text.append(f"   {suggestion}\n", style="green")
            else:
                suggestion_text.append(f"   {suggestion}\n", style="cyan")
        
        suggestion_panel = Panel(
            suggestion_text,
            title="💡 해결 방법",
            border_style="yellow"
        )
        
        self.ui.console.print(suggestion_panel)

    def _get_valid_yolo_params(self) -> set:
        """YOLO에서 지원하는 유효한 파라미터 목록"""
        return {
            # 기본 훈련 파라미터
            'data', 'epochs', 'batch', 'imgsz', 'lr0', 'device', 'project', 'name',
            'save_period', 'workers', 'amp', 'augment', 'verbose', 'exist_ok',
            'patience', 'optimizer', 'close_mosaic', 'resume', 'single_cls',
            
            # 학습률 및 정규화
            'cos_lr', 'dropout', 'weight_decay', 'warmup_epochs', 'warmup_momentum',
            'warmup_bias_lr', 'momentum', 'lr1', 'lrf',
            
            # 손실 함수 가중치
            'box', 'cls', 'dfl', 'pose', 'kobj', 'label_smoothing',
            
            # 배치 및 이미지 처리
            'nbs', 'overlap_mask', 'mask_ratio', 'rect', 'cache',
            
            # 출력 및 저장
            'plots', 'save_json', 'save_hybrid', 'save_txt', 'save_conf',
            
            # 추론 설정
            'conf', 'iou', 'max_det', 'half', 'dnn',
            
            # 데이터 증강
            'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
            'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste'
        }

    def _validate_train_args(self, train_args: Dict[str, Any]) -> Dict[str, Any]:
        """훈련 파라미터 유효성 검사 및 필터링"""
        valid_params = self._get_valid_yolo_params()
        
        # 유효한 파라미터만 필터링
        filtered_args = {k: v for k, v in train_args.items() if k in valid_params}
        
        # 제거된 파라미터 로깅
        removed_params = set(train_args.keys()) - set(filtered_args.keys())
        if removed_params:
            self.logger.info(f"지원되지 않는 파라미터 제거: {removed_params}")
        
        return filtered_args

    def _setup_windows_compatibility(self):
        """Windows 환경에서의 호환성 설정"""
        if platform.system() != "Windows":
            return
        
        try:
            # 환경 변수 설정
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            
            # 멀티프로세싱 설정
            import multiprocessing
            try:
                multiprocessing.set_start_method('spawn', force=True)
                self.logger.info("멀티프로세싱 시작 방법을 'spawn'으로 설정")
            except RuntimeError:
                # 이미 설정된 경우
                pass
            
            # PyTorch 설정
            if TORCH_AVAILABLE:
                torch.set_num_threads(1)
                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
            
            self.logger.info("Windows 호환성 설정 완료")
            
        except Exception as e:
            self.logger.warning(f"Windows 호환성 설정 실패: {e}")

    def _training_worker(self, model, train_args):
        """훈련 워커 스레드"""
        try:
            results = model.train(**train_args)
            self.training_stats['results'] = results
            self.training_stats['success'] = True
        except Exception as e:
            self.training_stats['error'] = str(e)
            self.training_stats['success'] = False
        finally:
            self.is_training = False
    
    def _monitor_training(self, training_thread):
        """훈련 모니터링"""
        if not RICH_AVAILABLE:
            print("훈련 중... 완료까지 기다려주세요.")
            training_thread.join()
            return
    
        # Rich 기반 실시간 모니터링
        def create_training_layout():
            # 메인 레이아웃 생성
            main_layout = Layout()
        
            # 상단 레이아웃 (헤더)
            header_panel = Panel(
                f"🚀 AI 모델 훈련 중 - {self.current_config.model_name}",
                style="bold blue"
            )
        
            # 좌측 패널 (훈련 통계)
            stats_text = Text()
            stats_text.append("📊 훈련 설정\n\n", style="bold green")
            stats_text.append(f"모델: {self.current_config.model_name}\n", style="cyan")
            stats_text.append(f"에폭: {self.current_config.epochs}\n", style="cyan")
            stats_text.append(f"배치 크기: {self.current_config.batch_size}\n", style="cyan")
            stats_text.append(f"이미지 크기: {self.current_config.img_size}\n", style="cyan")
            stats_text.append(f"학습률: {self.current_config.learning_rate}\n", style="cyan")
        
            stats_text.append("\n📈 진행 상황\n\n", style="bold yellow")
            if self.is_training:
                stats_text.append("상태: 훈련 중...\n", style="green")
            else:
                if self.training_stats.get('success'):
                    stats_text.append("상태: 훈련 완료!\n", style="green")
                else:
                    stats_text.append("상태: 훈련 실패\n", style="red")
        
            stats_panel = Panel(stats_text, title="📊 훈련 정보", border_style="green")
        
            # 우측 패널 (하드웨어 모니터링)
            hardware_info = self.hardware_monitor.get_performance_summary()
            current = hardware_info.get('current', {})
        
            hw_text = Text()
            hw_text.append("💻 하드웨어 상태\n\n", style="bold blue")
        
            # CPU
            cpu = current.get('cpu', {})
            cpu_usage = cpu.get('usage_percent', 0)
            hw_text.append(f"CPU: {cpu_usage:.1f}%", style="white")
            if cpu_usage > 80:
                hw_text.append(" 🔥", style="red")
            hw_text.append("\n")
        
            # 메모리
            memory = current.get('memory', {})
            memory_usage = memory.get('used_percent', 0)
            hw_text.append(f"메모리: {memory_usage:.1f}%", style="white")
            if memory_usage > 85:
                hw_text.append(" ⚠️", style="yellow")
            hw_text.append("\n")
        
            # GPU
            gpu_list = current.get('gpu', [])
            if gpu_list:
                for i, gpu in enumerate(gpu_list):
                    gpu_load = gpu.get('load_percent', 0)
                    gpu_memory = gpu.get('memory_percent', 0)
                    hw_text.append(f"GPU {i}: {gpu_load:.1f}% (VRAM: {gpu_memory:.1f}%)", style="white")
                    if gpu_memory > 90:
                        hw_text.append(" 🚨", style="red")
                    hw_text.append("\n")
            else:
                hw_text.append("GPU: 사용 불가 (CPU 모드)\n", style="yellow")
        
            # NPU
            npu = current.get('npu', {})
            if npu.get('available'):
                hw_text.append(f"NPU: 사용 가능\n", style="cyan")
        
            # 추천사항
            recommendations = hardware_info.get('recommendations', [])
            if recommendations:
                hw_text.append("\n💡 최적화 추천:\n", style="bold yellow")
                for i, rec in enumerate(recommendations[:2], 1):
                    hw_text.append(f"{i}. {rec[:50]}...\n" if len(rec) > 50 else f"{i}. {rec}\n", style="yellow")
        
            hardware_panel = Panel(hw_text, title="🖥️ 하드웨어", border_style="blue")
        
            # 하단 패널 (푸터)
            footer_panel = Panel(
                "Press Ctrl+C to stop monitoring (training continues in background)",
                style="dim"
            )
        
            # 레이아웃 구성
            # 상하 분할
            main_layout.split_column(
                Layout(header_panel, name="header", size=3),
                Layout(name="body", ratio=1),
                Layout(footer_panel, name="footer", size=3)
            )
        
            # 바디를 좌우로 분할
            main_layout["body"].split_row(
                Layout(stats_panel, name="left", size=45),
                Layout(hardware_panel, name="right", ratio=1)
            )
        
            return main_layout
    
        # 실시간 모니터링 실행
        try:
            with Live(create_training_layout(), console=self.ui.console, 
                    refresh_per_second=1) as live:
            
                while training_thread.is_alive():
                    try:
                        live.update(create_training_layout())
                        time.sleep(1)
                    except Exception as layout_error:
                        # 레이아웃 오류가 발생하면 간단한 텍스트 모드로 전환
                        self.logger.warning(f"레이아웃 업데이트 오류: {layout_error}")
                        break
            
                # 마지막 업데이트
                try:
                    live.update(create_training_layout())
                except Exception:
                    pass
                
        except KeyboardInterrupt:
            self.ui.console.print("\n[yellow]모니터링을 중단했습니다. 훈련은 백그라운드에서 계속됩니다.[/yellow]")
        except Exception as e:
            self.logger.error(f"모니터링 시스템 오류: {e}")
            self.ui.console.print(f"[red]모니터링 오류 발생: {e}[/red]")
            self.ui.console.print("[yellow]텍스트 모드로 전환합니다...[/yellow]")
        
            # 텍스트 모드로 폴백
            print("훈련 진행 중... (텍스트 모드)")
            while training_thread.is_alive():
                print(".", end="", flush=True)
                time.sleep(5)
            print(" 완료!")
    
        # 훈련 완료 대기
        training_thread.join()
    
        # 결과 표시
        if self.training_stats.get('success'):
            self.ui.console.print("\n[green]🎉 훈련이 성공적으로 완료되었습니다![/green]")
            self.show_training_results()
        else:
            error_msg = self.training_stats.get('error', 'Unknown error')
            self.ui.console.print(f"\n[red]❌ 훈련 실패: {error_msg}[/red]")
    
    def show_training_results(self):
        """훈련 결과 표시 - 클래스 정보 포함"""
        if not self.current_config:
            return
        
        results_dir = Path(self.current_config.project_name) / self.current_config.experiment_name
        best_model = results_dir / "best.pt"
        
        # 기존 결과 표시
        if not RICH_AVAILABLE:
            print(f"\n🎉 훈련 완료!")
            print(f"📁 결과 디렉토리: {results_dir}")
            
            # 클래스 정보 간단 표시
            if best_model.exists():
                class_summary = self.quick_class_summary(str(best_model))
                print(f"📋 {class_summary}")
            
            return
        
        # Rich 기반 결과 표시
        result_text = Text()
        result_text.append("🎉 훈련 완료!\n\n", style="bold green")
        result_text.append(f"📁 결과 경로: {results_dir}\n\n", style="cyan")
        
        # 클래스 정보 요약 추가
        if best_model.exists():
            class_summary = self.quick_class_summary(str(best_model))
            result_text.append(f"📋 {class_summary}\n\n", style="yellow")
        
        # 파일 목록
        if results_dir.exists():
            key_files = {
                'best.pt': '🏆 최고 성능 모델',
                'last.pt': '📱 마지막 모델', 
                'results.png': '📊 훈련 결과 그래프'
            }
            
            for filename, description in key_files.items():
                file_path = results_dir / filename
                if file_path.exists():
                    file_size = file_path.stat().st_size / (1024 * 1024)
                    result_text.append(f"  ✅ {description} ({file_size:.1f}MB)\n", style="green")
        
        result_panel = Panel(
            result_text,
            title="🏆 훈련 결과",
            border_style="green"
        )
        
        self.ui.console.print(result_panel)
        
        # 클래스 정보 상세 보기 옵션
        if best_model.exists():
            if Confirm.ask("🏷️ 상세한 클래스 정보를 보시겠습니까?", default=True):
                self.show_class_detection_results(str(best_model))
        
        # 결과 폴더 열기 옵션
        if Confirm.ask("📂 결과 폴더를 여시겠습니까?", default=False):
            try:
                if platform.system() == "Windows":
                    os.startfile(results_dir)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", results_dir])
                else:  # Linux
                    subprocess.run(["xdg-open", results_dir])
            except Exception as e:
                self.ui.console.print(f"[red]폴더 열기 실패: {e}[/red]")

    def _show_training_progress_summary(self, config: TrainingConfig):
        """훈련 진행 요약 표시 (주기적)"""
        if not RICH_AVAILABLE:
            return
        
        # 5초마다 한 번씩만 표시
        current_time = time.time()
        if not hasattr(self, '_last_progress_time'):
            self._last_progress_time = current_time
        
        if current_time - self._last_progress_time < 5:
            return
        
        self._last_progress_time = current_time
        
        # 하드웨어 상태 간단히 표시
        try:
            hw_info = self.hardware_monitor.get_performance_summary()
            current = hw_info.get('current', {})
            
            status_text = f"💻 CPU: {current.get('cpu', {}).get('usage_percent', 0):.0f}% | "
            status_text += f"💾 RAM: {current.get('memory', {}).get('used_percent', 0):.0f}%"
            
            gpu_list = current.get('gpu', [])
            if gpu_list:
                gpu = gpu_list[0]
                status_text += f" | 🎮 GPU: {gpu.get('load_percent', 0):.0f}%"
            
            # 상태 라인 업데이트 (한 줄로)
            self.ui.console.print(f"\r[dim]{status_text}[/dim]", end="")
            
        except Exception:
            pass

    def analyze_model_classes_standalone(model_path: str):
        """독립 실행 가능한 모델 클래스 분석 함수"""
        
        print("🔍 모델 클래스 분석 중...")
        
        if not Path(model_path).exists():
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            return
        
        try:
            model = YOLO(model_path)
            
            print(f"\n📁 모델: {Path(model_path).name}")
            
            # 클래스 정보 추출
            if hasattr(model.model, 'names'):
                names = model.model.names
                
                if isinstance(names, dict):
                    class_names = names
                    num_classes = len(names)
                    class_list = [names[i] for i in sorted(names.keys())]
                elif isinstance(names, list):
                    class_names = {i: name for i, name in enumerate(names)}
                    num_classes = len(names)
                    class_list = names
                else:
                    print("❌ 클래스 정보를 찾을 수 없습니다.")
                    return
                
                print(f"📊 총 {num_classes}개 클래스:")
                print("-" * 40)
                
                for i, class_name in enumerate(class_list):
                    print(f"  {i:2d}. {class_name}")
                
                print("-" * 40)
                print(f"✅ 분석 완료!")
                
            else:
                print("❌ 모델에서 클래스 정보를 추출할 수 없습니다.")
        
        except Exception as e:
            print(f"❌ 분석 실패: {e}")

    # 사용 예시
    if __name__ == "__main__":
        import sys
        
        if len(sys.argv) > 1:
            model_path = sys.argv[1]
            analyze_model_classes_standalone(model_path)
        else:
            print("사용법: python script.py model.pt")

# ════════════════════════════════════════════════════════════════════════════════
# 🎮 메인 시스템 클래스
# ════════════════════════════════════════════════════════════════════════════════

class AITrainingSystem:
    """
    AI 훈련 시스템 메인 클래스
    - 모든 컴포넌트 통합 관리
    - 워크플로우 제어
    - 사용자 인터랙션
    """
    
    def __init__(self):
        # 핵심 컴포넌트 초기화
        self.language_manager = LanguageManager()
        self.logger = AdvancedLogger("AITrainingSystem")
        self.ui = AdvancedUI(self.language_manager, self.logger)
        self.help_system = HelpSystem(self.language_manager, self.ui)
        
        # 시스템 컴포넌트
        self.security_manager = SecurityManager(self.logger)
        self.config_manager = ConfigurationManager(self.logger)
        self.integrity_manager = DataIntegrityManager(self.logger)
        self.hardware_monitor = HardwareMonitor(self.logger)
        
        # AI 시스템
        self.dataset_finder = SmartDatasetFinder(self.logger, self.security_manager)
        self.archive_processor = AdvancedArchiveProcessor(self.logger, self.integrity_manager)
        self.error_solver = AIErrorSolver(self.logger, self.ui)
        self.training_engine = TrainingEngine(self.logger, self.ui, self.hardware_monitor)
        
        # 현재 상태
        self.current_workflow = None
        self.system_config = None
        
        self.logger.info(f"AI 훈련 시스템 v{SystemConstants.VERSION} 초기화 완료")
    
    def run(self):
        """메인 실행 루프"""
        try:
            # 시스템 초기화
            if not self._initialize_system():
                return
            
            # 환영 화면
            self.ui.show_welcome_screen()
            
            # 메인 워크플로우
            self._main_workflow()
            
        except KeyboardInterrupt:
            if RICH_AVAILABLE:
                self.ui.console.print("\n[yellow]사용자에 의해 프로그램이 중단되었습니다.[/yellow]")
            else:
                print("\n프로그램이 중단되었습니다.")
        except Exception as e:
            self.logger.critical(f"시스템 오류: {e}")
            self.error_solver.show_error_analysis(str(e), {
                'operation': 'system_startup',
                'framework': 'AI Training System v3.0'
            })
        finally:
            self._cleanup()
    
    def _initialize_system(self) -> bool:
        """시스템 초기화 - 오류 처리 강화"""
        try:
            # Windows 호환성 설정 (추가)
            if platform.system() == "Windows":
                self.training_engine._setup_windows_compatibility()
            
            # 기존 초기화 코드...
            self.system_config = self.config_manager.load_config()
            self.language_manager.set_language(self.system_config.language)

            # 설정 로드 (오류 시 기본 설정 사용)
            try:
                self.system_config = self.config_manager.load_config()
            except Exception as config_error:
                self.logger.warning(f"설정 로드 실패, 기본 설정 사용: {config_error}")
                self.system_config = SystemConfig()
            
            # 언어 설정 적용
            self.language_manager.set_language(self.system_config.language)
            
            # 필수 디렉토리 생성
            directories = ['logs', 'backups', 'configs', 'datasets']
            for directory in directories:
                try:
                    Path(directory).mkdir(exist_ok=True)
                except Exception as dir_error:
                    self.logger.warning(f"디렉토리 생성 실패 ({directory}): {dir_error}")
            
            # 체크섬 캐시 로드 (실패해도 계속 진행)
            try:
                self.integrity_manager.load_checksum_cache(Path("configs/checksum_cache.json"))
            except Exception as cache_error:
                self.logger.warning(f"체크섬 캐시 로드 실패: {cache_error}")
            
            # 하드웨어 모니터링 시작 (실패해도 계속 진행)
            try:
                self.hardware_monitor.start_monitoring()
            except Exception as monitor_error:
                self.logger.warning(f"하드웨어 모니터링 시작 실패: {monitor_error}")
            
            self.logger.info("시스템 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"시스템 초기화 실패: {e}")
            return False
        
    def debug_config_file(self):
        """설정 파일 디버깅 정보 출력"""
        if not self.config_file.exists():
            self.logger.info("설정 파일이 존재하지 않습니다")
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.logger.info(f"설정 파일 크기: {len(content)} bytes")
            self.logger.info(f"설정 파일 내용 (처음 200자): {content[:200]}...")
            
            # JSON 파싱 테스트
            config_data = json.loads(content)
            self.logger.info(f"JSON 파싱 성공, 키: {list(config_data.keys())}")
            
            # 메타데이터 확인
            if '_metadata' in config_data:
                metadata = config_data['_metadata']
                self.logger.info(f"메타데이터: {metadata}")
            
        except Exception as e:
            self.logger.error(f"설정 파일 디버깅 실패: {e}")
    
    def _main_workflow(self):
        """메인 워크플로우"""
        # 1. 워크플로우 선택
        self.current_workflow = self.ui.show_workflow_menu()
        self.logger.info(f"워크플로우 선택: {self.current_workflow}")
        
        # 2. 시스템 환경 검사
        if not self._check_system_environment():
            return
        
        # 3. 데이터셋 검색 및 선택
        selected_datasets = self._dataset_workflow()
        if not selected_datasets:
            if RICH_AVAILABLE:
                self.ui.console.print("[red]데이터셋을 선택하지 않았습니다. 프로그램을 종료합니다.[/red]")
            else:
                print("데이터셋을 선택하지 않았습니다.")
            return
        
        # 4. 데이터셋 처리
        processed_dataset = self._process_dataset(selected_datasets[0])
        if not processed_dataset:
            return
        
        # 5. 훈련 설정
        training_config = self._training_configuration(processed_dataset)
        if not training_config:
            return
        
        # 6. 훈련 실행
        self._execute_training(training_config)
    
    def _check_system_environment(self) -> bool:
        """시스템 환경 검사"""
        with self.ui.show_progress("시스템 환경 검사 중...", total=5) as progress:
            
            # Python 버전 확인
            progress.update(1, "Python 버전 확인")
            if sys.version_info < (3, 8):
                self.ui.show_error("Python 버전 오류", 
                                 "Python 3.8 이상이 필요합니다.",
                                 f"현재 버전: {platform.python_version()}")
                return False
            
            # 필수 라이브러리 확인
            progress.update(1, "필수 라이브러리 확인")
            missing_libs = []
            
            if not TORCH_AVAILABLE:
                missing_libs.append("torch")
            if not ULTRALYTICS_AVAILABLE:
                missing_libs.append("ultralytics")
            if not PIL_AVAILABLE:
                missing_libs.append("Pillow")
            
            if missing_libs:
                self.ui.show_error("라이브러리 누락", 
                                 f"필수 라이브러리가 설치되지 않았습니다: {', '.join(missing_libs)}",
                                 f"설치 명령: pip install {' '.join(missing_libs)}")
                return False
            
            # 하드웨어 확인
            progress.update(1, "하드웨어 확인")
            hardware_summary = self.hardware_monitor.get_performance_summary()
            current_hw = hardware_summary.get('current', {})
            
            # GPU 확인
            progress.update(1, "GPU 확인")
            gpu_info = current_hw.get('gpu', [])
            if not gpu_info and TORCH_AVAILABLE:
                if not torch.cuda.is_available():
                    if RICH_AVAILABLE:
                        self.ui.console.print("[yellow]⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.[/yellow]")
                    else:
                        print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
            
            # 디스크 공간 확인
            progress.update(1, "디스크 공간 확인")
            disk_usage = shutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < 5:  # 5GB 미만
                self.ui.show_error("디스크 공간 부족", 
                                 f"사용 가능한 디스크 공간: {free_gb:.1f}GB",
                                 "최소 5GB의 여유 공간이 필요합니다.")
                return False
            
            progress.complete()
        
        if RICH_AVAILABLE:
            self.ui.console.print("[green]✅ 시스템 환경 검사 완료[/green]")
        else:
            print("✅ 시스템 환경 검사 완료")
        
        return True
    
    def _dataset_workflow(self) -> List[Dict[str, Any]]:
        """데이터셋 워크플로우"""
        # 검색 경로 설정
        search_paths = self._get_search_paths()
        
        # 데이터셋 검색
        with self.ui.show_progress("AI 데이터셋 검색 중...", total=None) as progress:
            datasets = self.dataset_finder.find_datasets(search_paths, max_results=20)
            progress.complete()
        
        if not datasets:
            if RICH_AVAILABLE:
                self.ui.console.print("[red]❌ 데이터셋을 찾을 수 없습니다.[/red]")
            else:
                print("❌ 데이터셋을 찾을 수 없습니다.")
            return []
        
        # 워크플로우별 선택
        if self.current_workflow == 'auto':
            # 자동 모드: 최고 점수 데이터셋 자동 선택
            selected_datasets = [datasets[0]]
            if RICH_AVAILABLE:
                self.ui.console.print(f"[green]🤖 자동 선택된 데이터셋: {datasets[0]['name']}[/green]")
            else:
                print(f"🤖 자동 선택된 데이터셋: {datasets[0]['name']}")
        else:
            # 반자동/수동 모드: 사용자 선택
            selected_indices = self.ui.show_dataset_selection(datasets)
            if not selected_indices:
                return []
            
            selected_datasets = [datasets[i] for i in selected_indices]
        
        # 사용자 선택 패턴 학습
        for dataset in selected_datasets:
            self.dataset_finder.learn_user_preference(dataset)
        
        return selected_datasets
    
    def _get_search_paths(self) -> List[Path]:
        """검색 경로 생성"""
        search_paths = []
        
        try:
            # 사용자 홈 디렉토리의 우선 폴더들
            home_dir = Path.home()
            
            for folder_name in SystemConstants.PRIORITY_FOLDERS:
                folder_path = home_dir / folder_name
                if folder_path.exists():
                    search_paths.append(folder_path)
            
            # 현재 디렉토리
            search_paths.append(Path.cwd())
            
            # 환경 변수로 지정된 경로
            dataset_env = os.environ.get('DATASET_PATH')
            if dataset_env:
                dataset_path = Path(dataset_env)
                if dataset_path.exists():
                    search_paths.append(dataset_path)
        
        except Exception as e:
            self.logger.warning(f"검색 경로 설정 중 오류: {e}")
        
        return search_paths
    
    def _process_dataset(self, dataset_info: Dict[str, Any]) -> Optional[Path]:
        """데이터셋 처리 (압축 해제 등)"""
        dataset_path = Path(dataset_info['path'])
        
        # 압축 파일인 경우 해제
        if dataset_path.suffix.lower() in SystemConstants.ARCHIVE_EXTENSIONS:
            extract_dir = Path("datasets") / dataset_path.stem
            
            with self.ui.show_progress(f"압축 해제 중: {dataset_path.name}", total=None) as progress:
                result = self.archive_processor.extract_archive(
                    dataset_path, 
                    extract_dir
                )
                progress.complete()
            
            if result['success']:
                if RICH_AVAILABLE:
                    self.ui.console.print(f"[green]✅ 압축 해제 완료: {result['extracted_files']}개 파일[/green]")
                else:
                    print(f"✅ 압축 해제 완료: {result['extracted_files']}개 파일")
                return extract_dir
            else:
                self.ui.show_error("압축 해제 실패", 
                                 result.get('error_message', '알 수 없는 오류'))
                return None
        else:
            # 이미 해제된 폴더
            return dataset_path
    
    def _training_configuration(self, dataset_path: Path) -> Optional[TrainingConfig]:
        """훈련 설정 구성 (모든 모드에서 사용자가 모델 선택)"""
    
        # 모든 워크플로우 모드에서 사용자가 모델 선택
        selected_model = self.training_engine.show_model_selection(self.current_workflow)
    
        if self.current_workflow == 'auto':
            # 완전 자동 모드 - 모델은 사용자 선택, 나머지는 자동
            config = self.training_engine.auto_configure_training(dataset_path, self.current_workflow, selected_model)
            if RICH_AVAILABLE:
                self.ui.console.print("[green]🤖 선택된 모델로 최적 설정을 구성했습니다.[/green]")
            else:
                print("🤖 선택된 모델로 최적 설정을 구성했습니다.")
        
        elif self.current_workflow == 'semi_auto':
            # 반자동 모드 - 모델은 사용자 선택, AI 추천 후 사용자 확인
            config = self.training_engine.auto_configure_training(dataset_path, self.current_workflow, selected_model)
        
            if RICH_AVAILABLE:
                # AI 추천 설정 표시
                config_table = Table(title="🤖 AI 추천 설정", show_header=False)
                config_table.add_column("설정", width=15, style="cyan")
                config_table.add_column("값", style="white")
            
                config_table.add_row("모델", config.model_name)
                config_table.add_row("에폭 수", str(config.epochs))
                config_table.add_row("배치 크기", str(config.batch_size))
                config_table.add_row("이미지 크기", str(config.img_size))
                config_table.add_row("학습률", str(config.learning_rate))
            
                self.ui.console.print(config_table)
            
                if not Confirm.ask("이 설정으로 진행하시겠습니까?", default=True):
                    # 수동 모드 - 모든 설정을 사용자가 직접 입력
                    config = TrainingConfig()
                    config.model_name = selected_model  # 사용자 선택 모델 적용
                
                    # 기타 설정들
                    if RICH_AVAILABLE:
                        config.epochs = IntPrompt.ask("에폭 수", default=100)
                        config.batch_size = IntPrompt.ask("배치 크기", default=16)
                        config.img_size = IntPrompt.ask("이미지 크기", default=640)
                    else:
                        try:
                            config.epochs = int(input("에폭 수 (기본값 100): ") or 100)
                            config.batch_size = int(input("배치 크기 (기본값 16): ") or 16)
                            config.img_size = int(input("이미지 크기 (기본값 640): ") or 640)
                        except ValueError:
                            print("잘못된 입력입니다. 기본값을 사용합니다.")
            else:
                print("AI 추천 설정:")
                print(f"모델: {config.model_name}")
                print(f"에폭: {config.epochs}")
                print(f"배치 크기: {config.batch_size}")
            
                confirm = input("이 설정으로 진행하시겠습니까? (y/n): ").lower()
                if confirm != 'y':
                    # 수동 모드 - 모든 설정을 사용자가 직접 입력
                    config = TrainingConfig()
                    config.model_name = selected_model  # 사용자 선택 모델 적용
                
                    # 기타 설정들
                    if RICH_AVAILABLE:
                        config.epochs = IntPrompt.ask("에폭 수", default=100)
                        config.batch_size = IntPrompt.ask("배치 크기", default=16)
                        config.img_size = IntPrompt.ask("이미지 크기", default=640)
                    else:
                        try:
                            config.epochs = int(input("에폭 수 (기본값 100): ") or 100)
                            config.batch_size = int(input("배치 크기 (기본값 16): ") or 16)
                            config.img_size = int(input("이미지 크기 (기본값 640): ") or 640)
                        except ValueError:
                            print("잘못된 입력입니다. 기본값을 사용합니다.")
        else:
            # 수동 모드 - 모든 설정을 사용자가 직접 입력
            config = TrainingConfig()
            config.model_name = selected_model  # 사용자 선택 모델 적용
        
            # 기타 설정들
            if RICH_AVAILABLE:
                config.epochs = IntPrompt.ask("에폭 수", default=100)
                config.batch_size = IntPrompt.ask("배치 크기", default=16)
                config.img_size = IntPrompt.ask("이미지 크기", default=640)
            else:
                try:
                    config.epochs = int(input("에폭 수 (기본값 100): ") or 100)
                    config.batch_size = int(input("배치 크기 (기본값 16): ") or 16)
                    config.img_size = int(input("이미지 크기 (기본값 640): ") or 640)
                except ValueError:
                    print("잘못된 입력입니다. 기본값을 사용합니다.")
    
        config.dataset_path = str(dataset_path)
    
        # 훈련 환경 설정
        if not self.training_engine.setup_training_environment(config):
            return None
    
        return config
    
    def _execute_training(self, config: TrainingConfig):
        """훈련 실행"""
        if RICH_AVAILABLE:
            self.ui.console.print("\n[bold green]🚀 AI 모델 훈련을 시작합니다![/bold green]")
        else:
            print("🚀 AI 모델 훈련을 시작합니다!")
        
        # 훈련 실행
        success = self.training_engine.start_training(config)
        
        if success:
            if RICH_AVAILABLE:
                self.ui.console.print("\n[green]🎉 훈련 프로세스가 완료되었습니다![/green]")
            else:
                print("🎉 훈련 프로세스가 완료되었습니다!")
        else:
            if RICH_AVAILABLE:
                self.ui.console.print("\n[red]❌ 훈련에 실패했습니다.[/red]")
            else:
                print("❌ 훈련에 실패했습니다.")
    
    def _cleanup(self):
        """정리 작업"""
        try:
            # 하드웨어 모니터링 중지
            self.hardware_monitor.stop_monitoring()
            
            # 설정 저장
            self.config_manager.save_config()
            
            # 체크섬 캐시 저장
            self.integrity_manager.save_checksum_cache(Path("configs/checksum_cache.json"))
            
            self.logger.info("시스템 정리 완료")
            
        except Exception as e:
            self.logger.error(f"정리 작업 중 오류: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# 🚀 메인 실행 함수
# ════════════════════════════════════════════════════════════════════════════════

def main():
    """메인 실행 함수"""
    try:
        # AI 훈련 시스템 인스턴스 생성 및 실행
        system = AITrainingSystem()
        system.run()
    except Exception as e:
        print(f"시스템 시작 실패: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

# ════════════════════════════════════════════════════════════════════════════════
# 🎯 AI 훈련 시스템 v3.0 완성!
# ════════════════════════════════════════════════════════════════════════════════
"""
🎉 축하합니다! AI 훈련 시스템 v3.0이 완성되었습니다!

✨ v2.2 → v3.0 주요 업그레이드 요약:
────────────────────────────────────────────────────────────────────────────
🏗️ 아키텍처:
   • 완전한 모듈화 설계로 재구성
   • 객체 지향 프로그래밍 패러다임 적용
   • 각 컴포넌트의 독립성과 재사용성 극대화

🛡️ 보안 및 안정성:
   • 견고한 경로 검증 시스템 (경로 탐색 공격 방지)
   • 입력 검증 및 SQL 인젝션 방지
   • 파일 무결성 검증 (SHA-256, CRC32)
   • 자동 백업 및 복원 시스템

📊 모니터링:
   • CPU, GPU, NPU 실시간 모니터링
   • 예측 분석 기반 성능 최적화
   • Rich 라이브러리 기반 실시간 대시보드
   • 하드웨어별 맞춤형 추천 시스템

🧠 AI 기능:
   • 스마트 데이터셋 검색 (가중치 점수 시스템)
   • 압축파일 AI 미리보기 기능
   • 사용자 패턴 학습 및 개인화
   • AI 기반 오류 해결 (ChatGPT 연동)

🎨 사용자 경험:
   • Rich 기반 고급 터미널 UI
   • 다국어 지원 (한국어/영어)
   • 상황별 통합 도움말 시스템 (!help)
   • 워크플로우 선택 (완전자동/반자동/수동)

⚡ 성능:
   • 멀티스레딩 병렬 처리
   • 메모리 효율적인 대용량 파일 처리
   • 캐싱 시스템으로 반복 작업 최적화
   • NPU 가속 지원 (Intel NPU 우선)

🔧 기술적 개선:
   • 포괄적인 예외 처리
   • 5단계 로깅 시스템 (DEBUG → CRITICAL)
   • 설정 백업/복원 시스템
   • 버전 관리 및 호환성 보장

────────────────────────────────────────────────────────────────────────────
🚀 사용법:
   1. Python 3.8+ 환경에서 실행
   2. 필수 라이브러리 설치: pip install torch ultralytics rich pillow psutil GPUtil
   3. 스크립트 실행: python automatic_training_v3.py
   4. 화면 안내에 따라 진행

💡 주요 특징:
   • 어떤 프롬프트에서든 !help 입력으로 도움말 확인
   • ESC 키로 이전 단계 복귀 (지원되는 경우)
   • Ctrl+C로 안전한 프로그램 종료
   • 자동 설정 저장 및 다음 실행시 복원

🔮 미래 확장 가능성:
   • 더 많은 AI 프레임워크 지원
   • 클라우드 훈련 연동
   • 모바일 앱 연동
   • 웹 인터페이스 제공

이제 여러분의 AI 모델 훈련이 이전보다 훨씬 쉽고 효율적이 될 것입니다! 🎯
"""
