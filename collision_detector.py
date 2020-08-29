from math import cos, sin, pi
import matplotlib.pyplot as plt
from math import sqrt, ceil, atan2
import random
import matplotlib.animation as animation


def quadratic_solver(a, b, c):
    if a == 0: return -c/b

    start = b / (2 * a)
    rootsq = start**2 - c / a

    if rootsq < 0: return ()
    if rootsq == 0: return (-start,)

    root = sqrt(rootsq)
    x1 = -start + root
    x2 = -start - root
    return (x1, x2)


def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def sqdistance(x1, y1, x2, y2):
    a = x1 - x2
    b = y1 - y2
    return a**2 + b**2


def cart_to_polar(vx, vy):
    s = sqrt(vx**2 + vy**2)
    d = atan2(vy, vx)
    return (s, d)


def polar_to_cart(s, d):
    vx = s * cos(d)
    vy = s * sin(d)
    return (vx, vy)


def rotate_vector(vx, vy, angle):
    size, direction = cart_to_polar(vx, vy)
    direction += angle
    return polar_to_cart(size, direction)


class Circle(object):
    def __init__(self, x, y, r, m=1, d=0, speed=0):
        self.y = y
        self.x = x
        self.r = r
        self.m = m
        self.vx, self.vy = polar_to_cart(speed, d)

    def update_position(self, t=0.1):
        self.x = self.x + t * self.vx
        self.y = self.y + t * self.vy

    def check_collision(self, otherCircle):
        csq = (self.r + otherCircle.r)**2
        dsq = sqdistance(self.x, self.y, otherCircle.x, otherCircle.y)
        return dsq <= csq

    @staticmethod
    def handle_collision(c1, c2, energy_keep_fraction=0.9):
        angle_to_rotate = atan2(c2.y - c1.y, c2.x - c1.x)

        # Rotate circle positions and speeds so that x-axis is on
        # line between circle centers

        c1_vx, c1_vy = rotate_vector(c1.vx, c1.vy, -angle_to_rotate)
        c2_vx, c2_vy = rotate_vector(c2.vx, c2.vy, -angle_to_rotate)

        # Remove velocity of center mass and only look at the disposable energy

        v_cm = (c1.m * c1_vx + c2.m * c2_vx)/(c1.m + c2.m)

        c1_vx -= v_cm
        c2_vx -= v_cm

        # Figure out how hard each circle gets pushed during collisions

        if c1_vx < 0 and c2_vx > 0:
            return

        v_factor = energy_keep_fraction**0.5
        c1_vx *= -v_factor
        c2_vx *= -v_factor

        # Re-add velocity of center mass

        c1_vx += v_cm
        c2_vx += v_cm

        # Rotate circle positions and speeds back to original angles

        c1.vx, c1.vy = rotate_vector(c1_vx, c1_vy, angle_to_rotate)
        c2.vx, c2.vy = rotate_vector(c2_vx, c2_vy, angle_to_rotate)

    def is_in_bin(self, bin):
        xPair, yPair = bin
        xLow, xHigh = xPair
        yLow, yHigh = yPair

        circlePoints = [(self.x, self.y), (self.x - self.r, self.y), (self.x + self.r, self.y),
                        (self.x, self.y - self.r), (self.x, self.y + self.r)]

        for x, y in circlePoints:
            if xLow <= x <= xHigh and yLow <= y <= yHigh: return True

        cornerPoints = [(xLow, yLow), (xLow, yHigh),
                        (xHigh, yLow), (xHigh, yHigh)]

        rsq = self.r**2
        for x, y in cornerPoints:
            if sqdistance(x, y, self.x, self.y) <= rsq: return True

        return False

    def get_plot_points(self, numPoints=1000):

        spacing = (2 * pi/(numPoints-1))   # 360 degrees is 2 * pi radians
        thetapoints = [0 + i * spacing for i in range(numPoints)]

        xpoints = [(cos(theta) * self.r) + self.x for theta in thetapoints]
        ypoints = [(sin(theta) * self.r) + self.y for theta in thetapoints]

        return xpoints, ypoints


class World(object):
    def __init__(self, l, w):
        self.l = l
        self.w = w

    def make_circle(self, moving=False):
        maxr = -1
        while maxr < 0:
            x = random.uniform(0, self.l)
            y = random.uniform(0, self.w)
            maxr = min((self.l - x), (self.w - y), y, x)

            for circle in self.circles:
                d = sqrt(sqdistance(x, y, circle.x, circle.y))
                maxr = min(maxr, d - circle.r)
                if maxr < 0:
                    break

        r = random.uniform(0, maxr)
        m = 2*pi*r

        if moving:
            speed = 5*random.uniform(0, sqrt(self.l**2 + self.w**2))
            direction = random.uniform(0, 2*pi)
        else:
            speed = 0
            direction = 0

        return Circle(x,y,r, m=m, speed=speed, d=direction)

    def populate(self, amountCircle, moving=False):
        self.circles = []
        # self.circles = [self.make_circle(moving=moving) for i in range(amountCircle)]
        for i in range(amountCircle):
            newCircle = self.make_circle(moving=moving)
            self.circles.append(newCircle)

    def update_world(self, t=0.1, energy_loss_fraction=0.5):
        for i, circle in enumerate(self.circles):
            circle.update_position(t=t)

        collisions = self.find_all_collisions()
        for i, j in collisions:
            Circle.handle_collision(self.circles[i], self.circles[j],
                                    energy_keep_fraction=1-energy_loss_fraction)
        for circle in self.circles:
            if (circle.x < circle.r and circle.vx < 0) or \
               (circle.r + circle.x > self.l and circle.vx > 0):
                circle.vx = -circle.vx
            if (circle.y < circle.r and circle.vy < 0) or \
               (circle.r + circle.y > self.w and circle.vy > 0):
                circle.vy = -circle.vy

    def plot(self, simulate=None, time_step=1e-3, energy_loss_fraction=0.5):

        # TODO: Update this to do collisions more physically, not just reverse
        # directions

        fig, ax = plt.subplots()
        ax.set_xlim(0, self.l)
        ax.set_ylim(0, self.w)

        pltPnts = [ax.scatter(*circle.get_plot_points(), s=1) for circle in self.circles]

        if simulate is None:
            plt.show()
            return

        def update(stepnum):
            self.update_world(t=time_step, energy_loss_fraction=energy_loss_fraction)
            for circle, pltPnt in zip(self.circles, pltPnts):
                xp, yp = circle.get_plot_points()
                pltPnt.set_offsets(list(zip(xp, yp)))
            return pltPnts

        anim = animation.FuncAnimation(fig, update, frames=ceil(simulate/time_step),
                                       interval=5, blit=False, repeat=False)
        return anim

    def find_collisions(self, circles, collisions={}):
        # circles is a list of indices from self.circles
        # collisions is a dictionary that maps index pairs (i,j) to booleans

        for i in circles:
            for j in circles:
                if i >= j or ((i, j) in collisions): continue
                c = self.circles[i]
                otherCircle = self.circles[j]
                collisions[(i, j)] = c.check_collision(otherCircle)

        return collisions

    def split_world(self, xSteps, ySteps):

        xStepSize = self.l/xSteps
        yStepSize = self.w/ySteps
        xyBins = [((i*xStepSize, (i+1)*xStepSize),
                   (j*yStepSize, (j+1)*yStepSize))
                  for i in range(xSteps) for j in range(ySteps)]
        xyBins = {bin: [] for bin in xyBins}
        for i, circle in enumerate(self.circles):
            for bin, circleList in xyBins.items():
                if circle.is_in_bin(bin):
                    circleList.append(i)
        return xyBins

    def find_all_collisions(self, xSteps=2, ySteps=2):

        xyBins = self.split_world(xSteps, ySteps)
        collisions = dict()
        for circleList in xyBins.values():
            collisions = self.find_collisions(circleList, collisions=collisions)

        return [k for k, v in collisions.items() if v]

        xyBins = self.split_world(xSteps, ySteps)
        collisions = dict()
        for circleList in xyBins.values():
            collisions = self.find_collisions(circleList, collisions=collisions)

        return [k for k, v in collisions.items() if v]
