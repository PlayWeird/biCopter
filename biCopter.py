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


# Class that defines what gets calculated every pyglet dt
# sim_dt = number of loops per second
# pixels_per_meter = space representation conversion factor
# copter = instance of a vehicle in the simulation.
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
        self.copter.control_update(dt)
        self.copter.physics_update(dt)
        self.copter.draw()
        glPopMatrix()


class Copter:
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

        # These prop speeds perfectly counter gravity.
        self.prop_speeds = (sqrt(-self.gravity * self.mass / (2 * self.prop_conversion_factor)),
                            sqrt(-self.gravity * self.mass / (2. * self.prop_conversion_factor)))

        # Generalized mass matrix.
        self.M = np.matrix([
            [self.mass, 0, 0],
            [0, self.mass, 0],
            [0, 0, self.Icm]])

        # Maps squared prop speeds to force in body frame.
        self.H = np.matrix([[0.0, 0.0],
                            [self.prop_conversion_factor, self.prop_conversion_factor],
                            [-self.body_length * self.prop_conversion_factor,
                             self.body_length * self.prop_conversion_factor]])

        self.pid = PidController((10.0, 1.0, 10.0), (0.0, 0.0), (self.gravity, 20.0))
        self.start_time = time.time()

    def physics_update(self, dt):
        c, s = cos(self.q[2]), sin(self.q[2])
        Rvb = np.asarray([[c, s, 0],
                          [-s, c, 0],
                          [0, 0, 1]])

        prop_speed_squared = [vi * vi for vi in self.prop_speeds]
        c = np.asarray(prop_speed_squared).reshape((2, 1))

        F = np.matmul(self.H, c)

        # Defined from Body Frame
        q_b_dot_dot = np.matmul(np.linalg.inv(self.M), F)

        # Force of gravity from vehicle frame
        g_v = np.matrix([0, self.gravity, 0]).T
        g_b = np.matmul(Rvb, g_v)

        q_b_dot_dot += g_b
        q_v_dot_dot = np.matmul(Rvb.T, q_b_dot_dot)

        self.q_dot += q_v_dot_dot * dt
        self.q += self.q_dot * dt

    def vertical_control_helper(self, ):
        pass

    def control_update(self, dt):
        target_altitude = -1.9 * sin(time.time() - self.start_time)
        self.draw_box(target_altitude)

        measurement = self.q[1]
        effort = self.pid.get_effort(target_altitude, measurement, dt)
        base_speed = sqrt(-self.gravity * self.mass / (2 * self.prop_conversion_factor))
        prop_speed = base_speed + effort
        self.prop_speeds = (prop_speed, prop_speed)

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

    # draw target
    def draw_box(self, height):
        height = height / 2
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                             ('v2f', [-10, height,
                                      -10, height - 0.01,
                                      10, height - 0.01,
                                      10, height]),
                             ('c3B', [45, 29, 250] * 4)
                             )


class PidController():
    def __init__(self, gains=(0.0, 0.0, 0.0),
                 initial_condition=(0.0, 0.0),
                 effort_bounds=(0.0, 0.0),
                 integral_threshold=0.25):
        self.k_p, self.k_i, self.k_d = gains
        self.e_prev, self.i_prev = initial_condition
        self.min_effort, self.max_effort = effort_bounds
        self.integration_threshold = integral_threshold

    def get_effort(self, target, measurement, dt):
        e = target - measurement
        p = e
        i = self.i_prev + ((e + self.e_prev) * dt / 2.0)
        d = (e - self.e_prev) / dt

        if not (-self.integration_threshold < e < self.integration_threshold):
            i = 0.0

        self.e_prev = e
        self.i_prev = i

        effort = self.k_p * p + self.k_i * i + self.k_d * d
        effort = max(min(effort, self.max_effort), self.min_effort)

        return effort


if __name__ == '__main__':
    main()
