"""ZMQ 이미지 수신 공통 모듈.

송신측 가정: 한 메시지(또는 멀티파트의 마지막 프레임)가 JPEG/PNG 인코딩된 바이트.
송신측이 PUB이면 --mode sub, PUSH면 pull, REP(요청/응답)면 req 를 사용.
"""
import cv2
import numpy as np


class ZmqImageSource:
    def __init__(self, host="192.168.123.164", port=55555, mode="sub", bind=False, timeout_ms=1000):
        import zmq
        if mode not in ('sub', 'pull', 'req') or not 1 <= int(port) <= 65535 or timeout_ms <= 0:
            raise ValueError('ZMQ mode, port 또는 timeout 설정을 확인하세요.')
        self.zmq = zmq
        self.ctx = zmq.Context.instance()
        self.mode = mode
        self.bind, self.timeout_ms = bind, int(timeout_ms)
        self.addr = f"tcp://{'*' if bind else host}:{port}"
        self.sock = self._socket()

    def _socket(self):
        zmq = self.zmq
        sock = self.ctx.socket({'sub': zmq.SUB, 'pull': zmq.PULL, 'req': zmq.REQ}[self.mode])
        try:
            sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVHWM, 8)
            if self.mode == 'sub':
                sock.setsockopt(zmq.SUBSCRIBE, b'')
            # CONFLATE is unsuitable for topic + image multipart messages.
            (sock.bind if self.bind else sock.connect)(self.addr)
            return sock
        except Exception:
            sock.close(0)
            raise

    def read(self):
        """다음 프레임(BGR ndarray)을 반환. 타임아웃 시 None."""
        try:
            if self.mode == "req":
                self.sock.send(b"")
            parts = self.sock.recv_multipart()
        except self.zmq.Again:
            if self.mode == "req":  # REQ는 타임아웃 후 소켓 재생성 필요
                self._reset_req()
            return None
        if self.mode != 'req':
            for _ in range(7):
                try:
                    parts = self.sock.recv_multipart(flags=self.zmq.NOBLOCK)
                except self.zmq.Again:
                    break
        if not parts or not parts[-1]:
            raise ValueError('빈 ZMQ 이미지 메시지입니다.')
        buf = np.frombuffer(parts[-1], dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('ZMQ 데이터를 JPEG/PNG 이미지로 디코딩할 수 없습니다.')
        return frame

    def _reset_req(self):
        self.sock.close(0)
        self.sock = self._socket()

    def close(self):
        self.sock.close(0)
