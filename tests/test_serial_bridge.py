"""Most COM <-> TCP: przekazywanie bajtow, lista dostepu, sprzatanie portu."""

from __future__ import annotations

import socket
import threading
from collections import deque

import pytest

from lerobot_mp.robot.serial_bridge import serve, split_listen

TIMEOUT = 5.0


class LoopbackLink:
    """Udaje port szeregowy: oddaje przy odczycie to, co dostal przy zapisie.

    Wystarczy do sprawdzenia, ze most przepuszcza bajty w obie strony - a przy
    okazji odwzorowuje ksztalt wymiany z serwem (pytanie -> odpowiedz).
    """

    def __init__(self) -> None:
        self._buf: deque[int] = deque()
        self._cv = threading.Condition()
        self.closed = False

    @property
    def in_waiting(self) -> int:
        with self._cv:
            return len(self._buf)

    def read(self, size: int = 1) -> bytes:
        with self._cv:
            # Skonczone czekanie, jak w prawdziwym porcie z `timeout=` - inaczej
            # watek mostu nie mialby jak zauwazyc, ze ma sie zatrzymac.
            self._cv.wait_for(lambda: self._buf or self.closed, timeout=0.05)
            out = bytes(self._buf.popleft() for _ in range(min(size, len(self._buf))))
        return out

    def write(self, data: bytes) -> int:
        with self._cv:
            self._buf.extend(data)
            self._cv.notify_all()
        return len(data)

    def close(self) -> None:
        with self._cv:
            self.closed = True
            self._cv.notify_all()


class RunningBridge:
    """Most wystartowany na wolnym porcie, gotowy do polaczenia w tescie."""

    def __init__(self, **kwargs: object) -> None:
        self.links: list[LoopbackLink] = []
        self.stop = threading.Event()
        self._ready = threading.Event()
        self.port = 0

        def open_link() -> LoopbackLink:
            link = LoopbackLink()
            self.links.append(link)
            return link

        def on_ready(port: int) -> None:
            self.port = port
            self._ready.set()

        self._thread = threading.Thread(
            target=serve,
            args=(open_link, "127.0.0.1", 0),
            kwargs={"stop": self.stop, "on_ready": on_ready, **kwargs},
            daemon=True,
        )

    def __enter__(self) -> "RunningBridge":
        self._thread.start()
        assert self._ready.wait(TIMEOUT), "most nie wystartowal"
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        self._thread.join(timeout=TIMEOUT)

    def connect(self) -> socket.socket:
        conn = socket.create_connection(("127.0.0.1", self.port), timeout=TIMEOUT)
        conn.settimeout(TIMEOUT)
        return conn


def test_bridge_carries_bytes_in_both_directions():
    with RunningBridge() as bridge, bridge.connect() as conn:
        conn.sendall(b"\xff\xff\x01\x02\x01\xfb")
        assert conn.recv(64) == b"\xff\xff\x01\x02\x01\xfb"


def test_bridge_opens_the_port_only_when_a_client_arrives():
    with RunningBridge() as bridge:
        assert bridge.links == []
        with bridge.connect() as conn:
            conn.sendall(b"x")
            conn.recv(16)
            assert len(bridge.links) == 1


def test_bridge_closes_the_port_after_the_client_leaves():
    """Rozlaczenie musi oddac urzadzenie - inaczej kolejny klient zastanie zajete."""
    with RunningBridge() as bridge:
        with bridge.connect() as conn:
            conn.sendall(b"x")
            conn.recv(16)
        link = bridge.links[0]
        deadline = threading.Event()
        for _ in range(100):
            if link.closed:
                break
            deadline.wait(0.05)
        assert link.closed


def test_bridge_serves_the_next_client_after_the_previous_one():
    with RunningBridge() as bridge:
        for expected in (b"a", b"b"):
            with bridge.connect() as conn:
                conn.sendall(expected)
                assert conn.recv(16) == expected
        assert len(bridge.links) == 2


def read_until_closed(conn: socket.socket) -> bytes:
    """Odczyt konczacy sie zamknieciem polaczenia, niezaleznie od systemu.

    Zamkniecie gniazda z nieodczytanymi danymi w buforze Windows kwituje
    RST-em, a nie czystym FIN - i to raz jako 10054 (reset), raz jako 10053
    (abort), zaleznie od tego, co zdazylo sie wyslac. Dla testu to ten sam
    wynik: rozmowy nie ma.
    """
    try:
        return conn.recv(16)
    except OSError:
        return b""


def test_bridge_rejects_addresses_outside_allow():
    """Most rusza fizycznym ramieniem, wiec lista dostepu ma faktycznie odcinac."""
    with RunningBridge(allow=("10.11.12.13",)) as bridge:
        with bridge.connect() as conn:
            conn.sendall(b"x")
            assert read_until_closed(conn) == b""
        assert bridge.links == []


@pytest.mark.parametrize(
    "value, expected",
    [
        ("5555", ("127.0.0.1", 5555)),
        ("0.0.0.0:5555", ("0.0.0.0", 5555)),
        ("192.168.1.50:9000", ("192.168.1.50", 9000)),
    ],
)
def test_split_listen_understands_the_accepted_forms(value, expected):
    assert split_listen(value) == expected
