"""
LED light pattern like Google Home
"""


import apa102
import time
import threading
try:
    import queue as Queue
except ImportError:
    import Queue as Queue


class Pixels:
    PIXELS_N = 3
    BRIGHTNESS = 20

    wakeup_colors = [0, 64, 0, 0, 0, 0, 0, 0, 0]
    detected_colors = [64, 32, 0, 0, 0, 0, 0, 0, 0]
    confirmed_colors = [64, 0, 0, 0, 0, 0, 0, 0, 0]
    ota_colors = [64, 0, 0, 0, 0, 0, 0, 0, 0]
    no_network_colors = [64, 64, 0, 0, 0, 0, 0, 0, 0]
    recording_colors = [64, 0, 0, 0, 0, 0, 0, 0, 0]

    def __init__(self):
        self.basis = [0] * 3 * self.PIXELS_N
        self.basis[0] = 3
        self.basis[3] = 3
        self.basis[4] = 3
        self.basis[7] = 3

        self.colors = [0] * 3 * self.PIXELS_N
        self.dev = apa102.APA102(num_led=self.PIXELS_N)

        self.next = threading.Event()
        self.queue = Queue.Queue()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def wakeup(self, direction=0):
        def f():
            self._wakeup(direction)

        self.next.set()
        self.queue.put(f)

    def listen(self):
        self.next.set()
        self.queue.put(self._listen)

    def think(self):
        self.next.set()
        self.queue.put(self._think)

    def speak(self):
        self.next.set()
        self.queue.put(self._speak)

    def off(self):
        self.next.set()
        self.queue.put(self._off)

    def ota(self):
        self.next.set()
        self.queue.put(self._ota)

    def _run(self):
        while True:
            func = self.queue.get()
            func()

    def _wakeup(self, direction=0):
        for i in range(1, 5):
            colors = [25 * v for v in self.basis]
            self.write(colors)
            time.sleep(0.01)

        self.colors = colors

    def _listen(self):
        for i in range(1, 25):
            colors = [i * v for v in self.basis]
            self.write(colors)
            time.sleep(0.01)

        self.colors = colors

    def _think(self):
        colors = self.colors

        self.next.clear()
        while not self.next.is_set():
            colors = colors[3:] + colors[:3]
            self.write(colors)
            time.sleep(0.2)

        t = 0.1
        for i in range(0, 5):
            colors = colors[3:] + colors[:3]
            self.write([(v * (4 - i) / 4) for v in colors])
            time.sleep(t)
            t /= 2

        # time.sleep(0.5)

        self.colors = colors

    def _speak(self):
        colors = self.colors
        gradient = -1
        position = 24

        self.next.clear()
        while not self.next.is_set():
            position += gradient
            self.write([(v * position / 24) for v in colors])

            if position == 24 or position == 4:
                gradient = -gradient
                time.sleep(0.2)
            else:
                time.sleep(0.01)

        while position > 0:
            position -= 1
            self.write([(v * position / 24) for v in colors])
            time.sleep(0.01)

        # self._off()

    def _off(self):
        print ([0] * 3 * self.PIXELS_N)
        self.write([0] * 3 * self.PIXELS_N)

    def _on(self):
        self.write(self.wakeup_colors)
        time.sleep(0.01)
    def on(self):
        self.next.set()
        self.queue.put(self._on)

    def detected(self):
        self.write(self.detected_colors)
        time.sleep(0.01)

        self.colors = self.detected_colors


    def confirmed(self):

        self.write(self.confirmed_colors)
        time.sleep(0.01)

        self.colors = self.confirmed_colors

    def _ota(self):

        self.next.clear()
        while not self.next.is_set():
            self.write(self.ota_colors)
            time.sleep(0.1)
            self.off()

        self.colors = self.ota_colors

    def write(self, colors):
        for i in range(self.PIXELS_N):
            self.dev.set_pixel(i, int(colors[3*i]), int(colors[3*i + 1]), int(colors[3*i + 2]), self.BRIGHTNESS)

        self.dev.show()


pixels = Pixels()


if __name__ == '__main__':

    print (pixels.basis)
    print (pixels.colors)
    pixels.ota()
    while True:

        try:
            #pixels.ota()
            time.sleep(0.01)
        except KeyboardInterrupt:
            break


    pixels.off()
    time.sleep(1)
