import pyglet
from pyglet.gl import *
from math import sin, cos, pi
import random

def main():
    window = PendulumWindow()
    pyglet.clock.schedule_interval(window.myTick, window.dtExpected)
    pyglet.app.run()

class PendulumWindow(pyglet.window.Window): 
    def __init__(self):
        super(PendulumWindow, self).__init__()
        self.dtExpected = 1/60.0
        self.arm = RevoluteJoint(
            pos = (self.width//2, 3*self.height//4 ),
            length = 280.0
        )

        self.resetSim()
        self.g = -4.0
        
        self.pid = PIDController((370.282, 50.0, 40.0)).pid
        self.goal = -45
        self.goal_width = 0.5
        self.goal_length = 500.0

    def myTick(self, dt):
        self.t += dt
        torque = self.arm.get_torque(self.g)
        torque += self.pid(self.goal, self.arm.theta, dt).next()
        
        self.arm.update(torque, dt)
        self.clear()
        self.draw_goal()
        self.arm.draw_joint()

    def draw_goal(self):
        glPushMatrix()
        glTranslatef(self.arm.x,self.arm.y,0)
        glRotatef(self.goal+180, 0,0,1)
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
            ('v2f', [-self.goal_width/2.0, 0,
                self.goal_width/2.0, 0,
                self.goal_width/2.0, self.goal_length,
                -self.goal_width/2.0, self.goal_length]),
            ('c3B', [0, 255, 0]*4)
        )
        glPopMatrix()

    def resetSim(self):
        self.t = 0.0
        self.arm.theta = random.uniform(-45, 45)
        self.arm.omega = random.uniform(-5,5)

class Joint:
    def __init__(self, pos, length, mass):
        self.x, self.y = pos
        self.length = length
        self.mass = mass
        self.toDeg = pi/180.0

class RevoluteJoint(Joint):
    def __init__(self, pos, length, mass= 1, width = 10):
        Joint.__init__(self, pos, length, mass)
        self.joint_width = width
        self.theta = 0.0
        self.omega = 0.0
        self.pendulum_width = 5.0

    def draw_joint(self,):
        glPushMatrix()
        glTranslatef(self.x,self.y,0)
        glRotatef(self.theta+180, 0,0,1)
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
            ('v2f', [-self.pendulum_width/2.0, 0,
                self.pendulum_width/2.0, 0,
                self.pendulum_width/2.0, self.length,
                -self.pendulum_width/2.0, self.length]),
            ('c3B', [100, 0, 200]*4)
        )
        glPopMatrix()

    def get_torque(self, g):
        return self.mass*self.length*g*sin(self.theta*self.toDeg)

    def update(self, torque, dt):
        self.omega += torque*dt
        self.theta += self.omega*dt
        self.bound_theta()

    def bound_theta(self):
        if self.theta > 180:
            self.theta -= 360
        elif self.theta < -180:
            self.theta += 360

class PIDController:

    def __init__(self, pid_constants, inital_cond = (0.0, 0.0), max_torque = 1000.0):
        #PID values
        self.Kp, self.Ki, self.Kd = pid_constants
        self.Eprev, self.Iprev = inital_cond
        self.max_torque = max_torque

    def pid(self, goal, val, dt):
        while 1:

            ### pid calculations happend here ###


            ### end pid calculations ###

            # Bound output
            out = -(p + i + d)
            out = min(max(out, -self.max_torque), self.max_torque)

            # Calculate input for the system
            yield out

if __name__ == '__main__':
    main()