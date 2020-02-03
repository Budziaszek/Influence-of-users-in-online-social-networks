import sys
import time


class ProgressBar:
    unitColor = '\033[1;36m'
    endColor = '\033[1;37m '
    current = 0

    def __init__(self, count):
        self.count = count
        self.start = time.time()
        self.next()

    def next(self, i=1):
        incre = int(100.0 / self.count * self.current)
        end = time.time()
        hours, rem = divmod(end - self.start, 3600)
        minutes, seconds = divmod(rem, 60)
        if self.current != self.count - 1:
            sys.stdout.write('\r' + '|%s%s%s%s| %d%%' % (self.unitColor, '\033[7m' + ' ' * incre
                                                         + ' \033[27m', self.endColor, ' ' * (100 - incre), incre))
            sys.stdout.write("\ttime: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), int(seconds)))
            sys.stdout.flush()
            self.current += i

    def finish(self):
        sys.stdout.write('\n')
        end = time.time()
        hours, rem = divmod(end - self.start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\ttime: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), int(seconds)))
