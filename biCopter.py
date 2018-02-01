import pyglet
from pyglet.gl import *
import math
from math import cos, sin, pi, sqrt
import numpy as np
import time


def main():
    window = SimWindow()
    glClearColor(.5, .5, .5, 1.0)
    pyglet.clock.schedule_interval(window.my_tick, window.sim_dt)
    pyglet.app.run()


class SimWindow(pyglet.window.Window):
    def __init__(self):
        super(SimWindow, self).__init__(800, 600)
        self.sim_dt = 1.0 / 60.0
        self.pixels_per_meter = 200.0
        self.copter = Copter(q=np.matrix([0.0, 0.0, 0.0]).T)

    def my_tick(self, dt):
        self.clear()
        glPushMatrix()
        glTranslatef(self.width / 2.0, self.height / 2.0, 0)
        glScalef(self.pixels_per_meter, self.pixels_per_meter, self.pixels_per_meter)
        self.copter.update(dt)
        self.copter.draw()
        glPopMatrix()


class Copter():
    def __init__(self, body_length=0.25, mass=50.0, q=np.zeros((3, 1), np.float64)):
        # Appearance Variables
        self.body_length = body_length
        self.body_height = 5.0 / 200.0
        self.motor_size = 15.0 / 200.0
        self.prop_length = 25.0 / 200.0
        self.mass_radius = 11.0 / 200.0
        # Physics Variables
        self.mass = mass
        self.q = q  # Vehicle frame
        self.q_dot = np.zeros((3, 1), np.float64)  # Body frame
        self.Icm = (self.mass * self.body_length * 2.0) / 12.0
        self.prop_conversion_factor = 2.0
        self.gravity = -9.8
        self.prop_speeds = (sqrt(-self.gravity * self.mass / (2 * self.prop_conversion_factor)),
                            sqrt(-self.gravity * self.mass / (2. * self.prop_conversion_factor)))
        # self.prop_speeds = (0,0)

        self.M = np.matrix([
            [self.mass, 0, 0],
            [0, self.mass, 0],
            [0, 0, self.Icm]])

    def update(self, dt):
        c, s = cos(self.q[2]), sin(self.q[2])
        Rvb = np.asarray([[c, s, 0],
                          [-s, c, 0],
                          [0, 0, 1]])

        # Calculate Translational Accelerations
        effort = np.zeros((3, 1))
        prop_thrust = [self.prop_conversion_factor * vi * vi for vi in self.prop_speeds]
        torque = (prop_thrust[1] - prop_thrust[0]) * self.body_length
        F = sum(prop_thrust)

        effort[1], effort[2] = F, torque

        # Defined from Body Frame
        q_b_dot_dot = np.matmul(np.linalg.inv(self.M), effort)

        # Force of gravity from vehicle frame
        g_v = np.matrix([0, self.gravity, 0]).T
        g_b = np.matmul(Rvb, g_v)

        q_b_dot_dot += g_b
        q_v_dot_dot = np.matmul(Rvb.T, q_b_dot_dot)

        self.q_dot += q_v_dot_dot * dt
        self.q += self.q_dot * dt

    def draw(self):
        # Translate coordinates to center
        glPushMatrix()
        glTranslatef(self.q[0] / 2.0, self.q[1] / 2.0, 0)
        glRotatef(math.degrees(self.q[2]), 0, 0, 1)

        # draw copter body
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                             ('v2f', [-self.body_length, -self.body_height,
                                      self.body_length, -self.body_height,
                                      self.body_length + self.motor_size / 2, self.body_height,
                                      -self.body_length - self.motor_size / 2, self.body_height]),
                             ('c3B', [66, 244, 100] * 4)
                             )

        # draw motors
        glPushMatrix()
        glTranslatef(self.body_length, self.body_height, 0)
        self.draw_motor()
        glPopMatrix()
        glPushMatrix()
        glTranslatef(-self.body_length, self.body_height, 0)
        self.draw_motor()
        glPopMatrix()

        # draw center of mass
        glPushMatrix()
        self.draw_mass_helper(True)
        glRotatef(90, 0, 0, 1)
        self.draw_mass_helper(False)
        glRotatef(90, 0, 0, 1)
        self.draw_mass_helper(True)
        glRotatef(90, 0, 0, 1)
        self.draw_mass_helper(False)
        glPopMatrix()
        glPopMatrix()

    def draw_mass_helper(self, black):
        number_of_triangles = 12
        angle = (2 * pi) / number_of_triangles
        color = [0, 0, 0]
        if not black:
            color = [255, 255, 255]

        pyglet.graphics.draw(5, pyglet.gl.GL_TRIANGLE_FAN,
                             ('v2f', [0, 0,
                                      self.mass_radius * cos(0), self.mass_radius * sin(0),
                                      self.mass_radius * cos(angle), self.mass_radius * sin(angle),
                                      self.mass_radius * cos(angle * 2), self.mass_radius * sin(angle * 2),
                                      self.mass_radius * cos(angle * 3), self.mass_radius * sin(angle * 3)]),
                             ('c3B', color * 5)
                             )

    def draw_motor(self):
        # draw motor box
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                             ('v2f', [-self.motor_size / 2.0, 0,
                                      self.motor_size / 2.0, 0,
                                      self.motor_size / 5.0, self.motor_size,
                                      -self.motor_size / 5.0, self.motor_size]),
                             ('c3B', [255, 229, 0] * 4)
                             )

        # draw prop pin

        pin_width = 0.005
        pin_height = 0.05
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                             ('v2f', [-pin_width, self.motor_size,
                                      pin_width, self.motor_size,
                                      pin_width, self.motor_size + pin_height,
                                      -pin_width, self.motor_size + pin_height]),
                             ('c3B', [0, 0, 0] * 4)
                             )

        # draw prop
        prop_height = 0.01
        prop_offset = 0.03
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                             ('v2f', [-self.prop_length, self.motor_size + prop_offset,
                                      self.prop_length, self.motor_size + prop_offset,
                                      self.prop_length, self.motor_size + prop_offset + prop_height,
                                      -self.prop_length, self.motor_size + prop_offset + prop_height]),
                             ('c3B', [255, 0, 0] * 4)
                             )


if __name__ == '__main__':
    main()
