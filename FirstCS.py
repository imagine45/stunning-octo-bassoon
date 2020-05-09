import random
class Circle(object):
    def __init__(self,x,y,r):
        self.y = y
        self.x = x
        self.r = r
    def check_collision(self,otherCircle):
        a = self.x - otherCircle.x
        b = self.y - otherCircle.y
        c = self.r + otherCircle.r
        var = (a**2 + b**2 <= c**2)
        return var

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

    def find_all_collisions(self, circles, collisions = {}):
        # circles is a dictionary that maps index
        # in self.circles to the circle objects
        # collisions is a dictionary that maps index pairs (i,j) to booleans
        
        for i, c in circles.items():
            for j, otherCircle in circles.items():
                if i > j or ((i,j) in collisions): continue
                collisions[(i,j)] = c.check_collision(otherCircle)

        return collisions
        
        
