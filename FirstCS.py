import random

def distance(x1, y1, x2, y2):
    a = x1 - x2
    b = y1 - y2
    return a**2 + b**2

class Circle(object):
    def __init__(self,x,y,r):
        self.y = y
        self.x = x
        self.r = r

    def check_collision(self,otherCircle):
        c = self.r + otherCircle.r
        return sqdistance(self.x, self.y, otherCircle.x, otherCircle.y) <= c**2

    def is_in_bin(self, bin):
        xPair, yPair = bin
        xLow, xHigh = xPair
        yLow, yHigh = yPair

        circlePoints = [(self.x, self.y), (self.x - r, self.y), (self.x + r, self.y),
                        (self.x, self.y - r), (self.x, self.y + r)]

        for x, y in circlePoints:
            if xLow <= x <= xHigh and yLow <= y <= yHigh: return True

        cornerPoints = [(xLow, yLow), (xLow, yHigh),
                        (xHigh, yLow), (xHigh, yHigh)]

        rsq = self.r**2
        for x, y in cornerPoints:
            if sqdistance(x, y, self.x, self.y) <= rsq: return True

        return False




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

    def find_collisions(self, circles, collisions = {}):
        # circles is a list of indices from self.circles
        # collisions is a dictionary that maps index pairs (i,j) to booleans

        for i in circles:
            for j in circles:
                if i > j or ((i,j) in collisions): continue
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


        if (x)
