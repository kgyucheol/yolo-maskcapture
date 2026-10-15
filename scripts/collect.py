"""1. 데이터 수집: ZMQ 스트림을 받아 data/raw/images 에 저장.

조작키 (미리보기 창):  s=1장 저장 | SPACE=자동저장 토글 | q/ESC=종료
예) python scripts/collect.py --host 192.168.0.10 --interval 0.5 --auto
    python scripts/collect.py --no-display --auto --interval 1 --max 500   # 헤드리스
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zmq_source import ZmqImageSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.123.164")
    ap.add_argument("--port", type=int, default=55555)
    ap.add_argument("--mode", default="sub", choices=["sub", "pull", "req"])
    ap.add_argument("--bind", action="store_true", help="connect 대신 bind (송신측이 connect 하는 경우)")
    ap.add_argument("--out", default="data/raw/images")
    ap.add_argument("--interval", type=float, default=0.5, help="자동저장 간격(초)")
    ap.add_argument("--auto", action="store_true", help="시작부터 자동저장")
    ap.add_argument("--max", type=int, default=0, help="최대 저장 장수 (0=무제한)")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = ZmqImageSource(args.host, args.port, args.mode, args.bind)
    print(f"[collect] {src.addr} ({args.mode}) -> {out}")

    auto, last_save, saved = args.auto, 0.0, 0

    def save(frame):
        nonlocal saved, last_save
        name = out / f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
        cv2.imwrite(str(name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
        last_save = time.time()
        print(f"saved {saved}: {name.name}")

    try:
        while not (args.max and saved >= args.max):
            frame = src.read()
            if frame is None:
                print("[collect] 프레임 대기 중...")
                continue
            if auto and time.time() - last_save >= args.interval:
                save(frame)
            if args.no_display:
                continue
            view = frame.copy()
            cv2.putText(view, f"{'AUTO' if auto else 'MANUAL'}  saved={saved}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if auto else (0, 200, 255), 2)
            cv2.imshow("collect", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                save(frame)
            elif key == ord(" "):
                auto = not auto
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        cv2.destroyAllWindows()
        print(f"[collect] 총 {saved}장 저장")


if __name__ == "__main__":
    main()
