import time
import numpy as np
from queue import Queue
from threading import Thread, RLock


def generate_seeds(n: int, low: int, high: int) -> np.ndarray:
    r = np.random.randint(low, high, (n,))
    seeds = np.sqrt(np.power(r, 2)-1)
    return seeds


class Intending:
    a: np.ndarray
    b: np.ndarray

    def mk_random(self, n: int, low: int, high: int):
        self.a = generate_seeds(n, low, high)
        self.b = generate_seeds(n, low, high)

    def generate(self, t: float) -> float:
        return 0.5 + 0.5 * np.tanh(np.sum(self.a * np.sin(t*self.b)))


class Subject:
    name: str
    intending: Intending

    def __init__(self, name: str = 'Subject'):
        self.intending = Intending()
        self.intending.mk_random(n=7, low=5, high=20)
        self.name = name


class RollingData:
    data = []
    queue = Queue()
    _lock = RLock()
    max_data_length: float = 10.0  # seconds

    def __init__(self):
        Thread(target=self._discard_loop, args=(), daemon=True).start()
        Thread(target=self._fetch_queue_loop, args=(), daemon=True).start()

    def peek(self, data_length: float = None) -> list:
        with self._lock:
            t = self.data[-1][0]
            if data_length is not None:
                t_start = t-data_length
            else:
                t_start = t-self.max_data_length
            return [e for e in self.data if e[0] > t_start]

    def _insert(self, t: float, g: float):
        with self._lock:
            self.data.append((t, g))

    def _fetch_queue_loop(self):
        while True:
            got = self.queue.get()
            t, g = got
            self._insert(t, g)

    def _discard(self, preserve_t: float):
        with self._lock:
            if len(self.data) == 0:
                return
            t = self.data[-1][0]
            self.data = [e for e in self.data if e[0] > t-preserve_t]

    def _discard_loop(self):
        while True:
            self._discard(preserve_t=self.max_data_length)
            time.sleep(self.max_data_length)
