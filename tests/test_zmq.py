"""The external socket is doubled because pyzmq is unavailable offline; image decoding stays real."""
import sys
import types

import cv2
import numpy as np


class Again(Exception):
    pass


class Socket:
    def __init__(self, payload):
        self.payload, self.options = payload, {}
        self.address = None

    def setsockopt(self, key, value):
        self.options[key] = value

    def bind(self, address):
        self.address = ('bind', address)

    def connect(self, address):
        self.address = ('connect', address)

    def recv_multipart(self, *args, **kwargs):
        if self.payload is None:
            raise Again()
        payload, self.payload = self.payload, None
        return payload

    def send(self, value):
        pass

    def close(self, *args):
        pass


def transport(monkeypatch, payloads):
    sockets = []
    def socket(kind):
        sock = Socket(payloads.pop(0) if payloads else None)
        sockets.append(sock)
        return sock
    context = types.SimpleNamespace(socket=socket)
    module = types.SimpleNamespace(Context=types.SimpleNamespace(instance=lambda: context), Again=Again,
        SUB=1, PULL=2, REQ=3, RCVTIMEO=4, SNDTIMEO=5, LINGER=6, SUBSCRIBE=7, RCVHWM=8, NOBLOCK=9)
    monkeypatch.setitem(sys.modules, 'zmq', module)
    return sockets


def test_topic_multipart_decodes_last_frame(monkeypatch):
    frame = np.full((4, 8, 3), 123, np.uint8)
    payload = cv2.imencode('.png', frame)[1].tobytes()
    transport(monkeypatch, [[b'camera/color', payload]])
    from zmq_source import ZmqImageSource
    source = ZmqImageSource()
    assert np.array_equal(source.read(), frame)
    source.close()


def test_req_timeout_reset_keeps_bind_and_custom_timeout(monkeypatch):
    sockets = transport(monkeypatch, [None, [cv2.imencode('.png', np.zeros((4, 8, 3), np.uint8))[1].tobytes()]])
    from zmq_source import ZmqImageSource
    source = ZmqImageSource(mode='req', bind=True, timeout_ms=73)
    assert source.read() is None
    assert sockets[-1].address == ('bind', 'tcp://*:55555')
    assert sockets[-1].options[4] == 73
    assert source.read().shape == (4, 8, 3)
    source.close()
