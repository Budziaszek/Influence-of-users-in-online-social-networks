import logging
import sys
import time
from math import ceil


class ProgressBar:
    unitColor = '\033[1;36m'
    endColor = '\033[1;37m '
    current = 0

    def __init__(self, start_communicate, finish_communicate, count):
        self.count = count
        self.start = time.time()
        self.start_communicate = start_communicate
        self.finish_communicate = finish_communicate
        self.next()

    @staticmethod
    def set_colors(unit_color, end_color):
        ProgressBar.unitColor = unit_color
        ProgressBar.endColor = end_color

    def next(self, i=1):
        if self.current == 0:
            logging.info(self.start_communicate)
        time.sleep(100)
        incre = int(ceil((100.0 / self.count * self.current)))
        end = time.time()
        hours, rem = divmod(end - self.start, 3600)
        minutes, seconds = divmod(rem, 60)
        if self.current != self.count - 1 and logging.root.level <= logging.INFO:
            sys.stdout.write('\r|%s%s%s%s| %d%%' %
                             (self.unitColor, '\033[7m' + ' ' * incre + ' \033[27m',
                              self.endColor, ' ' * (100 - incre), incre))
            sys.stdout.write("\ttime: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), int(seconds)))
            sys.stdout.flush()
            self.current += i

    def finish(self):
        sys.stdout.write('\n')
        if logging.root.level <= logging.INFO:
            print(self.finish_communicate)
