import numpy as np
import matplotlib.pyplot as plt
from random import randint

"""pyplot wrapper for easier use"""


class Line:
    def __init__(self, x, y, line_width=1, line_color=None, text=None, style='-',
                 axis_labels={'y': 'y', 'x': 'x'}, marker=None):
        if x is None:
            x = np.arange(len(y))
        self.x = x
        self.y = y
        self.width = line_width
        self.axis_labels = axis_labels
        self.marker = marker

        if line_color is None:

            random_hex = str(hex(randint(0x111111, 0xcccccc)))
            self.color = '#' + random_hex[2:]
        else:
            self.color = line_color

        self.text = text
        self.style = style

    def plot(self, fig=None, text_y=1, log=False, title='', labels=True):
        if fig is None:
            plt.figure()
            plt.grid(True)
            plt.ylabel(self.axis_labels['y'])
            plt.xlabel(self.axis_labels['x'])

        plt.title(title)

        max_y, min_y = self.y_range()

        plt.plot(self.x,
                 self.y,
                 self.color,
                 linewidth=self.width,
                 markersize=self.width * 3,
                 linestyle=self.style,
                 label=self.text,
                 marker=self.marker)
        # plt.text(self.x[0], text_y,
        #          self.text, color=self.color)

        if labels:
            plt.legend()

        if log:
            plt.yscale('log')

        if fig is None:
            plt.show()

    def y_range(self):
        return np.amin(self.y), np.amax(self.y)


class Point(Line):

    def __init__(self, x, y, line_width=1, line_color=None, text='', marker='.'):
        super().__init__(x, y, line_width, line_color, text, marker=marker)


class Function(Line):

    def __init__(self, x, func, line_width=1, line_color=None, text='', style='-'):
        y = [func(i) for i in x]
        super().__init__(x, y, line_width, line_color, text, style)


class Constant(Line):

    def __init__(self, x, c, line_width=1, line_color=None, text='', style='-'):
        x = [x[0], x[len(x) - 1]]
        y = [c] * len(x)
        super().__init__(x, y, line_width, line_color, text, style)


def plot_lines(lines, seps=None, grid_flag=True, log=False, title='',
               axis_labels={'y': 'y', 'x': 'x'}, labels=True):
    fig = plt.figure()
    plt.grid(grid_flag)
    plt.title(title)
    plt.ylabel(axis_labels['y'])
    plt.xlabel(axis_labels['x'])
    max_y = []
    min_y = []
    for line in lines:
        temp = line.y_range()
        min_y.append(temp[0])
        max_y.append(temp[1])

    min_y = np.amin(min_y)
    max_y = np.amax(max_y)

    count = 1
    y_range = max_y - min_y
    for line in lines:
        line.plot(fig=fig, text_y=0.5 * y_range * count / len(lines), labels=labels)
        count += 1

    if log:
        plt.yscale('log')

    if seps is not None:
        for s in seps:
            plt.plot([s, s], [min_y, max_y], 'r', linewidth=0.5)

    plt.show()
