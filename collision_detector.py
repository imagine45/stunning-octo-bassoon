from math import cos, sin, pi
import matplotlib.pyplot as plt
from math import sqrt, ceil, atan2
import random
import matplotlib.animation as animation


def quadratic_solver(a, b, c):

    ##Come back to this

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


class ReferenceFrame(object):

    def __init__(self, angle, *object_list, cm=True):
        self.object_list = object_list
        self.angle = angle
        self.x, self.y, self.vx, self.vy = 0, 0, 0, 0

##cm = Center of Mass

        if cm:
            m = sum(shape.m for shape in self.object_list) ##Adds the masses of all the objects
            self.x = sum(shape.m * shape.x for shape in self.object_list)/m ##^^^ except it multiplies by x of each shape and divides by mass
            self.y = sum(shape.m * shape.y for shape in self.object_list)/m ##^^^ but y
            self.vx = sum(shape.m * shape.vx for shape in self.object_list)/m ##^^^ but with vx
            self.vy = sum(shape.m * shape.vy for shape in self.object_list)/m ##^^^ but with vy

    def __enter__(self):
        for shape in self.object_list:
            shape.x, shape.y = rotate_vector(shape.x - self.x,
                                             shape.y - self.y,
                                             self.angle)
            shape.vx, shape.vy = rotate_vector(shape.vx - self.vx,
                                               shape.vy - self.vy,
                                               self.angle)

    def __exit__(self, type, value, traceback):
        for shape in self.object_list:
            shape.x, shape.y = rotate_vector(shape.x, shape.y, -self.angle)
            shape.x += self.x
            shape.y += self.y

            shape.vx, shape.vy = rotate_vector(shape.vx, shape.vy, -self.angle)
            shape.vx += self.vx
            shape.vy += self.vy

class Shape(object):
    def __init__(self, x, y, m=1, d=0, speed=0):
        self.x, self.y = x, y
        self.m = m
        self.vx, self.vy = polar_to_cart(speed, d)

    def update_position(self, t=0.1):
        self.x += t * self.vx
        self.y += t * self.vy

    def update_velocity(self, ax, ay, t=0.1):
        self.vx += t * ax
        self.vy += t * ay

    def check_collision(self, otherShape):
        raise NotImplementedError

    def is_in_bin(self, bin):
        raise NotImplementedError

    def get_plot_points(self, numPoints=1000):
        raise NotImplementedError


class Circle(Shape):
    def __init__(self, x, y, r, m=1, d=0, speed=0):
        super().__init__(x, y, m=m, d=d, speed=speed)
        self.r = r

    def check_collision(self, otherCircle):

##csq finds minimum sqdistance between two circles without colliding, dsq finds what the sqdistance is.

        csq = (self.r + otherCircle.r)**2
        dsq = sqdistance(self.x, self.y, otherCircle.x, otherCircle.y)
        return dsq <= csq

    @staticmethod
    def handle_collision(c1, c2, energy_keep_fraction=0.9):
        angle_to_rotate = atan2(c2.y - c1.y, c2.x - c1.x)

        # Need to rotate circle positions and speeds so
        # that x-axis is on line between circle centers

        with ReferenceFrame(-angle_to_rotate, c1, c2):
            if c1.vx > 0 and c2.vx < 0:
                v_factor = energy_keep_fraction**0.5
                c1.vx *= -v_factor
                c2.vx *= -v_factor

    def is_in_circle(self, x, y):
        """
        Given a point (x, y), returns whether or not the point is in this circle
        Inputs:
            x - x-coordinate of point
            y - y-coordinate of point

        Returns a boolean - True if point is in circle, False otherwise
        """
        return (sqdistance(x, y, self.x, self.y) <= self.r**2)

    def is_in_bin(self, bin):
        """
        Given a bin (an aligned rectangle in space), returns whether or not the
        circle is in the bin (i.e. whether it intersects the bin).

        Inputs:
            bin - a pair of pairs of numbers ((xLow, xHigh), (yLow, yHigh)) that
                  describe the ends of the rectangle in the x and y directions

        Returns a boolean - True if the circle intersects the bin, False otherwise
        """

        xPair, yPair = bin
        xLow, xHigh = xPair
        yLow, yHigh = yPair

        circlePoints = [(self.x, self.y),
                        (self.x - self.r, self.y),
                        (self.x + self.r, self.y),
                        (self.x, self.y - self.r),
                        (self.x, self.y + self.r)]

        for x, y in circlePoints:
            if xLow <= x <= xHigh and yLow <= y <= yHigh: return True

        cornerPoints = [(xLow, yLow),
                        (xLow, yHigh),
                        (xHigh, yLow),
                        (xHigh, yHigh)]

        for x, y in cornerPoints:
            if self.is_in_circle(x, y): return True

        return False

    def get_plot_points(self, numPoints=1000):
        """
        Given the amount of points to plot a circle, returns the x and y
        positions of those points.

        Inputs:
            numPoints - the amount of points you want in the circle
            (more points would mean a more fleshed out circle)
            Default is 1000 points

        Returns (xpoints, ypoints), the two of which are the lists of positions
        of each point.
        """
        spacing = (2 * pi/(numPoints-1))   # 360 degrees is 2 * pi radians
        thetapoints = [0 + i * spacing for i in range(numPoints)]

        xpoints = [(cos(theta) * self.r) + self.x for theta in thetapoints]
        ypoints = [(sin(theta) * self.r) + self.y for theta in thetapoints]

        return xpoints, ypoints

class Cannon(Shape):
    def __init__(self, x, y, orientation, world,
                 cannonball_radius = 30,
                 cannonball_speed = 100,
                 cannonball_frequency = 1,
                 tbe=2, explosionr=5):
        super().__init__(x,y,m=0,speed=0)
        self.orientation = orientation
        self.cannonball_frequency = cannonball_frequency
        self.cannonball_radius = cannonball_radius
        self.cannonball_speed = cannonball_speed
        self.time_to_next_cannonball = 0
        self.tbe = tbe
        self.explosionr = explosionr
        self.world = world
        world.cannons.append(self)

    def update_position(self, t=0.1):
        self.time_to_next_cannonball -= t
        if self.time_to_next_cannonball <= 0:
            #make a cannonball
            circle1 = ExplodingCircle(self.x, self.y, self.cannonball_radius,
                                      self.world, d=self.orientation,
                                      speed=self.cannonball_speed,
                                      tbe=self.tbe, explosionr=self.explosionr)
            #add cannonball  to the world
            self.world.circles.append(circle1)
            self.world.exploded.append(False)
            self.time_to_next_cannonball += self.cannonball_frequency


    def change_orientation(self, angle):
        self.orientiation += angle

class ExplodingCircle(Circle):
    def __init__(self, x, y, r, world, m=1, d=0, speed=0, tbe=2, explosionr=5):
        super().__init__(x, y, r, m=m, d=d, speed=speed)
        self.tbe = tbe
        self.explosionr = explosionr
        self.world = world

    def update_position(self, t=0.1):
        super().update_position(t=0.1)
        self.tbe -= t
        if self.tbe <= 0:
            self.world.track_explosion(self)


class World(object):
    def __init__(self, l, w, gravity=False, gravity_strength=10):
        """Inputs are length and width (l and w).
        Length is on the x axis and width is on the y axis."""
        self.l = l
        self.w = w
        self.gravity = [0, -gravity_strength if gravity else 0]
        self.cannons = []

    def make_circle(self, moving=False, type="ring", exploding_radius = 5,
                    exploding_time=None):
        """Creates a circle that is not immediately colliding with the edge of
        the world or another circle in the world.

            Inputs:
                moving - if you want your new circle to move or not.
                 Default is False.
                type - "ring" or "disc". Ring is like an outline of a circle,
                 and a disc is a filled in one. Default is "ring".
        """
        maxr = -1
        while maxr < 0:
            x = random.uniform(0, self.l) # Center of circle
            y = random.uniform(0, self.w)
            maxr = min((self.l - x), (self.w - y), y, x)
            # picks the minimum distance from the edge of the world.

            for circle in self.circles:
                d = sqrt(sqdistance(x, y, circle.x, circle.y))
                maxr = min(maxr, d - circle.r)
                if maxr < 0:
                    break

        r = random.uniform(0, maxr)
        if type == "ring":
            m = 2*pi*r # m = mass
        elif type == "disc":
            m = pi*r**2
        if moving:
            speed = 5*random.uniform(0, sqrt(self.l**2 + self.w**2))
            direction = random.uniform(0, 2*pi)
        else:
            speed = 0
            direction = 0
        if exploding_time is None:
            return Circle(x, y, r, m=m, speed=speed, d=direction)
        else:
            return ExplodingCircle(x, y, r, self,
                                   m=m, d=direction, speed=speed,
                                   tbe=exploding_time,
                                   explosionr=exploding_radius)

    def add_circle(self, moving=False, type="ring", exploding_radius = 5,
                   exploding_time=None):

        self.circles.append(self.make_circle(moving=moving, type=type,
                                             exploding_radius=exploding_radius,
                                             exploding_time=exploding_time))
        self.exploded.append(False)


    def populate(self, amountCircle, moving=False, type="ring"):
        """Creates an amount of new circles in the world to replace any old ones

        Inputs:
            amountCircle - How many circles you want to make.
            moving - Do you want your circles to move? Default is False.
            type - "ring" or "disc". Ring is like an outline of a circle,
             and a disc is a filled in one. Default is "ring". """
        self.circles = []
        self.exploded = []
        for i in range(amountCircle):
            self.add_circle(moving=moving, type=type)

    def update_world(self, t=0.1, energy_loss_fraction=0.5):

        """Updates the world to simulate motion once.

        Inputs:
            t - time between each update. Default = 0.1
            energy_loss_fraction - fraction of energy you lose every collision.
            Default = 0.5"""

        for cannon in self.cannons:
            cannon.update_position(t=t)

        for circle in self.circles:
            circle.update_position(t=t)
            circle.update_velocity(self.gravity[0], self.gravity[1], t=t)

        self.circles = [circle for i, circle in enumerate(self.circles)
                        if not self.exploded[i]]

        self.exploded = [False for circle in self.circles]

        collisions = self.find_all_collisions()
        for i, j in collisions:
            Circle.handle_collision(self.circles[i], self.circles[j],
                                    energy_keep_fraction=1-energy_loss_fraction)
        v_factor = (1-energy_loss_fraction)**0.5
        for circle in self.circles:
            if (circle.x < circle.r and circle.vx <= 0) or \
               (circle.r + circle.x > self.l and circle.vx > 0):
                circle.vx *= -v_factor
            if (circle.y < circle.r and circle.vy < 0) or \
               (circle.r + circle.y > self.w and circle.vy > 0):
                circle.update_velocity(self.gravity[0], -self.gravity[1], t=t)
                circle.vy *= -v_factor


    def track_explosion(self, exploder):
        explosion = Circle(exploder.x, exploder.y, exploder.r + exploder.explosionr)
        for i, circle in enumerate(self.circles):
            if circle.check_collision(explosion):
                self.exploded[i] = True



    def plot(self, simulate=None, time_step=1e-3, energy_loss_fraction=0.5):

        """Plots all points and animates the circles if you wish it.

        Inputs:
            simulate (float) - Whether or not you want your plot to simulate and if so how long
             you want it to run. Default is that the plot will not simulate.
            time_step - The amount of time between updates. Default is 1e-3.
            energy_loss_fraction - the fraction of energy lost every collision.
             Default is 0.5.

         If simulate is unspecified, then it will return nothing.

         If simulate is a number, it will return an animation.

        """

        fig, ax = plt.subplots() # Returns an overall figure object (contains all graphs you might make) and an axis object (just the one graph you actually care about)
        ax.set_xlim(0, self.l) # Sets start and end points on plot x-axis to match the world
        ax.set_ylim(0, self.w) # Sets start and end points on plot y-axis to match the world

        xl, yl = [], []
        for circle in self.circles:
            xp, yp = circle.get_plot_points()
            xl += xp
            yl += yp
        pltPnts = ax.scatter(xl, yl, s=1)

        if simulate is None: return

        def update(stepnum):
            self.update_world(t=time_step, energy_loss_fraction=energy_loss_fraction)
            xl, yl = [], []
            for circle in self.circles:
                xp, yp = circle.get_plot_points()
                xl += xp
                yl += yp
            pltPnts.set_offsets(list(zip(xl, yl))) # Updates graph (pltPnt) to have new x and y values
            return [pltPnts]

        anim = animation.FuncAnimation(fig, update, frames=ceil(simulate/time_step),
                                       interval=time_step*1e3, blit=False, repeat=False)
        return anim

    def find_collisions(self, circle_indices, collisions={}):
        # circles is a list of indices from self.circles
        # collisions is a dictionary that maps index pairs (i,j) to booleans
        """Adding new entries to collision dictionary. Checks first to see if
            what it's checking is already in the dictionary.

            Inputs:

                circle_indices - A list of indices from self.circles to check
                 collisions between specific circles.
                collisions - A dictionary that maps index pairs (i,j) to
                 booleans. Default is a blank dictionary.

            Returns the new and improved collisions dictionary.

            MODIFIES THE ORIGINAL DICTIONARY"""
        for i in circle_indices:
            for j in circle_indices:
                if i >= j or ((i, j) in collisions): continue #Checks to see if
                # comparisons have already been made, since you compare the
                #lower indices to all the higher ones beforehand, you don't need
                #to check again.
                c = self.circles[i]
                otherCircle = self.circles[j]
                collisions[(i, j)] = c.check_collision(otherCircle)

        return collisions

    def split_world(self, xSteps, ySteps):

        """Splits the world into bins and figures out which circles are in which
            bins.

            Inputs:
                xSteps - The amount of steps across the world horizontally. In
                 other words, the number of bins on the x axis.
                ySteps - The amount of steps across the world vertically. In
                 other words, the number of bins on the y axis.
            Returns a dictionary which contains the bins as keys and the circles
             within them in a list as values."""


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

        """Finds every collision within every bin.

            Inputs:
                xSteps - The amount of steps across the world horizontally. In
                 other words, the number of bins on the x axis.
                ySteps - The amount of steps across the world vertically. In
                 other words, the number of bins on the y axis.

            Returns every pair of circles that collided."""

        xyBins = self.split_world(xSteps, ySteps)
        collisions = {}
        for circleList in xyBins.values():
            collisions = self.find_collisions(circleList, collisions=collisions)

        return [k for k, v in collisions.items() if v]
