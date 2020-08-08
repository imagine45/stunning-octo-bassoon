from math import cos, sin, pi
import matplotlib.pyplot as plt
from math import sqrt, ceil, atan2
import random
import matplotlib.animation as animation
def quadratic_solver(a,b,c):

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

class Circle(object):
    def __init__(self,x,y,r,m=1,d=0,speed=0):
        self.y = y
        self.x = x
        self.r = r
        self.m = m
        self.vx, self.vy = polar_to_cart(speed, d)

    def update_position(self, t=0.1):
        self.x = self.x + t * self.vx
        self.y = self.y + t * self.vy

    def check_collision(self,otherCircle):
        c = self.r + otherCircle.r
        return sqdistance(self.x, self.y, otherCircle.x, otherCircle.y) <= c**2

    @staticmethod
    def handle_collision(c1, c2, energy_keep_fraction=0.9):

        init_v1 = (c1.vx, c1.vy)
        init_v2 = (c2.vx, c2.vy)

        r_diff = (c2.x - c1.x, c2.y - c1.y)

        old_E1 = 0.5 * c1.m * dot(init_v1, init_v1)
        old_E2 = 0.5 * c2.m * dot(init_v2, init_v2)
        r1_sqrdiff = 0.5 * c1.m * dot(r_diff, r_diff)
        r2_sqrdiff = 0.5 * c2.m * dot(r_diff, r_diff)

        v1dotrdiff = -0.5 * c1.m * 2 * dot(init_v1, r_diff)
        v2dotrdiff = 0.5 * c2.m * 2 * dot(init_v2, r_diff)

        k1 = quadratic_solver((r1_sqrdiff + (c1.m / c2.m)**2 * r2_sqrdiff),
                              (v1dotrdiff + c1.m / c2.m * v2dotrdiff),
                              (1-energy_keep_fraction) * (old_E1 + old_E2))

        try:
            assert len(k1) != 0
            k1 = max(k1)
            assert k1 > 0
        except:
            print(f"Circle 1 at ({c1.x},{c1.y}) with velocity ({c1.vx, c1.vy})")
            print(f"Circle 2 at ({c2.x},{c2.y}) with velocity ({c2.vx, c2.vy})")
            print(f"k1 is {k1}")
            return

        k2 = k1 * c1.m / c2.m

        c1.vx = c1.vx - k1 * r_diff[0]
        c1.vy = c1.vy - k1 * r_diff[1]

        c2.vx = c2.vx + k2 * r_diff[0]
        c2.vy = c2.vy + k2 * r_diff[1]


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

    def get_plot_points(self):
        spacing = ((self.x + self.r)-(self.x - self.r))/(1000-1)
        xpoints = [self.x - self.r + i * spacing for i in range(1000)]


        def yfunc(x):
            try:
                var = sqrt(self.r**2 - (x - self.x)**2)
            except:
                var = 0
            return [self.y + var, self.y - var]

        ypoints = sum([yfunc(x) for x in xpoints], [])
        xpoints = sum([[x,x] for x in xpoints], [])
        return xpoints, ypoints

    def get_even_plot_points(self, numPoints = 1000):
##        spacing = ((self.x + self.r)-(self.x - self.r))/(1000-1)
##        xpoints = [self.x - self.r + i * spacing for i in range(1000)]

        spacing = (2 * pi/(numPoints-1))   # 360 degrees is 2 * pi radians
        thetapoints = [0 + i * spacing for i in range(numPoints)]

        # Way 0

        xpoints = [(cos(theta) * self.r) + self.x for theta in thetapoints]
        ypoints = [(sin(theta) * self.r) + self.y for theta in thetapoints]

##        # Way 1
##
##        def xyfunc(theta):
##                x = (cos(theta) * self.r) + self.x
##                y = (sin(theta) * self.r) + self.y
##            return [x,y]
##
##        xypoints = [xyfunc(theta) for theta in thetapoints]
##
##
##
##        # One way
##
##        xpoints = []
##        ypoints = []
##        for a in xypoints:
##            xpoints.append(a[0])
##            ypoints.append(a[1])
##
##
##        # Other way
##
##        ypoints = [a[1] for a in xypoints]
##        xpoints = [a[0] for a in xypoints]
##
##
##        # Shortest code way
##        xpoints, ypoints = zip(*xypoints)

        return xpoints, ypoints




# def zip(*lists):
#
#     zippedlist = []
#     for i in range(lists[0]):
#         t = []
#         for list in lists:
#             t.append(list[i])
#         zippedlist.append(t)
def cart_to_polar(vx, vy):
    s = sqrt(vx**2 + vy**2)
    d = atan2(vy, vx)
    return (s, d)

def polar_to_cart(s, d):
    vx = s * cos(d)
    vy = s * sin(d)
    return (vx, vy)

def circle_collision(c1, c2):
    #Figure out x difference and y difference between their centers

    #Figure out the angle for the line between their centers (using x difference and y difference)

    xdifference = c1.x - c2.x
    ydifference = c1.y - c2.y

    angle_to_rotate = atan2(ydifference, xdifference)

    c1_s, c1_d = cart_to_polar(c1.vx, c1.vy)
    c2_s, c2_d = cart_to_polar(c2.vx, c2.vy)

    c1_d = c1_d - angle_to_rotate
    c2_d = c2_d - angle_to_rotate

    c1_vx, c1_vy = polar_to_cart(c1_s, c1_d)
    c2_vx, c2_vy = polar_to_cart(c2_s, c2_d)

    c1_vx = - c1_vx
    c2_vx = - c2_vx

    c1_s, c1_d = cart_to_polar(c1_vx, c1_vy)
    c2_s, c2_d = cart_to_polar(c2_vx, c2_vy)

    c1_d = c1_d + angle_to_rotate
    c2_d = c2_d + angle_to_rotate

    c1.vx, c1.vy = polar_to_cart(c1_s, c1_d)
    c2.vx, c2.vy = polar_to_cart(c2_s, c2_d)


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
            #circle_collision(self.circles[i], self.circles[j])
            Circle.handle_collision(self.circles[i], self.circles[j],
                                    energy_keep_fraction=1-energy_loss_fraction)
        for circle in self.circles:
            if ((circle.x < circle.r and circle.vx < 0) or
                (circle.r + circle.x > self.l and circle.vx > 0)):
                circle.vx = -circle.vx
            if ((circle.y < circle.r and circle.vy < 0) or
                (circle.r + circle.y > self.w and circle.vy > 0)):
                circle.vy = -circle.vy

    def plot(self, simulate=None, time_step=1e-3, energy_loss_fraction=0.5):

        # TODO: Update this to do collisions more physically, not just reverse
        # directions

        fig, ax = plt.subplots()
        ax.set_xlim(0, self.l)
        ax.set_ylim(0, self.w)

        pltPnts = [ax.scatter(*circle.get_even_plot_points()) for circle in self.circles]

        if simulate is None:
            plt.show()
            return

        def update(stepnum):
            self.update_world(t=time_step, energy_loss_fraction=energy_loss_fraction)
            for circle, pltPnt in zip(self.circles, pltPnts):
                xp, yp = circle.get_even_plot_points()
                pltPnt.set_offsets(list(zip(xp,yp)))
            return pltPnts

        anim = animation.FuncAnimation(fig, update, frames=ceil(simulate/time_step),
                                       interval=5, blit=False, repeat=False)
        return anim

    def find_collisions(self, circles, collisions = {}):
        # circles is a list of indices from self.circles
        # collisions is a dictionary that maps index pairs (i,j) to booleans

        for i in circles:
            for j in circles:
                if i >= j or ((i,j) in collisions): continue
                c = self.circles[i]
                otherCircle = self.circles[j]
                collisions[(i,j)] = c.check_collision(otherCircle)

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
