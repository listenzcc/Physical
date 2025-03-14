from typing import List
import time
import numpy as np
from queue import Queue
from threading import Thread, RLock


def generate_irrational_numbers(n: int, low: int, high: int) -> np.ndarray:
    '''
    Generate a numpy array of random irrational numbers.

    $k = \sqrt(r^2-1)$

    $r \in \mathbb{N}^+, low < r < high$

    :param n int: the number of random irrational numbers to generate.
    :param low int: the lower bound of the generation.
    :param high int: the upper bound of the generation.

    :return np.ndarray: a numpy array of random irrational numbers.
    '''
    r = np.random.randint(low, high, (n,))
    seeds = np.sqrt(np.power(r, 2)-1)
    return seeds


class SpeakWilling:
    a: np.ndarray
    b: np.ndarray

    def mk_random(self, n: int, low: int, high: int):
        self.a = generate_irrational_numbers(n, low, high)
        self.b = generate_irrational_numbers(n, low, high)

    def generate(self, t: float) -> float:
        '''
        Generate a new value based on the given time.

        :param t float: the time in seconds.

        :return float: the new value in the range of (0, 1).
        '''
        d = np.sum([self.fb(t+i*100) for i in range(4)])/4
        return d

    def fa(self, t):
        return 0.5 + 0.5 * np.tanh(7*np.sum(self.a * np.sin(t*np.sqrt(self.b))))

    def fb(self, t):
        d = np.sum([self.fa(t+i*10) for i in range(10)]) / 10
        return d


class Subject:
    name: str
    speak_willing: SpeakWilling

    def __init__(self, name: str = 'Subject'):
        self.speak_willing = SpeakWilling()
        self.speak_willing.mk_random(n=7, low=5, high=20)
        self.name = name


class BaseData:
    t: float
    g: float
    s: float
    name: str = 'Subject'

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            try:
                self.__setattr__(k, v)
            except:
                pass


class RollingData:
    data: List[BaseData] = []
    max_data_length: float = 10.0  # seconds

    _queue: Queue = None
    _lock: RLock = None

    def __init__(self):
        self._queue = Queue()
        self._lock = RLock()
        self.start_loops()

    def start_loops(self):
        Thread(target=self._discard_loop, args=(), daemon=True).start()
        Thread(target=self._fetch_queue_loop, args=(), daemon=True).start()

    def peek(self, t: float = None, length: float = None) -> List[BaseData]:
        '''
        Peek the data from the queue.
        The data range is (t-length, t].

        :param t float: the time in seconds.
        :param length float: the length of the data.

        :return list: a list of BaseData.
        '''
        with self._lock:
            if t is None:
                t = self.data[-1].t

            if length is None:
                length = self.max_data_length

            t_start = t - length
            return [e for e in self.data if e.t > t_start and e.t < t]

    def _safety_append(self, bd: BaseData):
        '''
        Safety append the inputs into the self.data.

        :param t float: the time in seconds.
        :param g float: the g value.
        :param name: the name of the subject.
        '''
        if bd.t < 1.0:
            bd.s = bd.g
        else:
            dt = 0.05
            num_subjects = 20
            m = 5
            n = 1.5

            ds = self.peek(bd.t, 0.3)
            print(set([e.name for e in ds]))
            print(set([e.t for e in ds]))

            a = np.sum([e.s for e in ds]) * dt / num_subjects
            ln = np.log(1+a/0.3) / np.log(2)

            ds = self.peek(bd.t, 0.25)
            c = np.sum([e.s for e in ds]) * dt / num_subjects

            ds = self.peek(bd.t-0.5, 0.25)
            d = np.sum([e.s for e in ds]) * dt / num_subjects

            psi = 0.5 + 0.5 * np.tanh(m*(2*(c-d) + n))
            # print(bd.t, bd.g, ln, psi)

            bd.s = bd.g * ln * psi + 0.001 * bd.g

        with self._lock:
            self.data.append(bd)

    def _fetch_queue_loop(self):
        while True:
            got = self._queue.get()
            self._safety_append(got)

    def _discard(self, preserve_t: float):
        with self._lock:
            if len(self.data) == 0:
                return
            t = self.data[-1].t
            self.data = [e for e in self.data if e.t > t-preserve_t]

    def _discard_loop(self):
        while True:
            self._discard(preserve_t=self.max_data_length)
            time.sleep(self.max_data_length)
