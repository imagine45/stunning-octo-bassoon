from math import cos, sin, pi
import matplotlib.pyplot as plt
from math import sqrt, ceil
import random

def sqdistance(x1, y1, x2, y2):
    a = x1 - x2
    b = y1 - y2
    return a**2 + b**2

class Circle(object):
    def __init__(self,x,y,r,d=0,speed=0):
        self.y = y
        self.x = x
        self.r = r
        if speed < 0:
            d = d + pi
            speed = -speed
        self.d = d + 2 * pi * ceil(-d/(2*pi))
        self.speed = speed
    def check_collision(self,otherCircle):
        c = self.r + otherCircle.r
        return sqdistance(self.x, self.y, otherCircle.x, otherCircle.y) <= c**2

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




def zip(*lists):

    zippedlist = []
    for i in range(lists[0]):
        t = []
        for list in lists:
            t.append(list[i])
        zippedlist.append(t)


class World(object):
    def __init__(self, l, w):
        self.l = l
        self.w = w

    def make_circle(self):
        x = random.uniform(0, self.l)
        y = random.uniform(0, self.w)
        maxr = min((self.l - x), (self.w - y), y, x)
        r = random.uniform(0, maxr)
        return Circle(x,y,r)

    def populate(self, amountCircle):
        self.circles = [self.make_circle() for i in range(amountCircle)]

    def plot_world(self):
        for c in self.circles:
            xpoints, ypoints = c.get_even_plot_points(numPoints=10)
            plt.scatter(xpoints, ypoints)
        plt.show()


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
