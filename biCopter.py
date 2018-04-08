import random

import pyglet
from pyglet.gl import *
from math import cos, sin, asin, pi, sqrt, degrees
import numpy as np


def main():
    window = SimWindow()
    glClearColor(.5, .5, .5, 1.0)
    pyglet.clock.schedule_interval(window.my_tick, window.sim_dt)
    pyglet.app.run()


# Class that defines what gets calculated every pyglet dt
# sim_dt = time between frames
# pixels_per_meter = space representation conversion factor
# copter = instance of a vehicle in the simulation.
class SimWindow(pyglet.window.Window):
    def __init__(self):
        super(SimWindow, self).__init__(1680, 1000)
        self.sim_dt = 1.0 / 60.0
        self.pixels_per_meter = 200.0
        self.copters = []
        # Allows multiple copter to be loaded at once
        for i in range(1):
            self.copters.append(Copter(i, q=np.matrix([random.uniform(-4.0, 4.0), random.uniform(-4.0, 4.0), 0.0]).T))

    # Runs every frame at rate dt
    def my_tick(self, dt):
        dt = self.sim_dt
        self.clear()
        glPushMatrix()
        glTranslatef(self.width / 2.0, self.height / 2.0, 0)
        glScalef(self.pixels_per_meter, self.pixels_per_meter, self.pixels_per_meter)
        for copter in self.copters:
            copter.control_update(dt)
            copter.physics_update(dt)
            copter.draw()
        glPopMatrix()

class Copter:
    def __init__(self, copter_id, body_length=0.25, mass=20.0, q=np.zeros((3, 1))):
        self.copter_id = copter_id
        self.target_alt = random.uniform(-4.0, 4.0)
        self.target_x = random.uniform(-4.0, 4.0)
        # Appearance Variables
        self.body_length = body_length
        self.body_height = 0.025
        self.motor_size = .075
        self.prop_length = .125
        self.mass_radius = .055
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

        # Generalized mass matrix (encapsulates mass and inertia.)
        self.M = np.matrix([
            [self.mass, 0, 0],
            [0, self.mass, 0],
            [0, 0, self.Icm]])

        # Maps squared prop speeds to force in body frame.
        self.H = np.matrix([[0.0, 0.0],
                            [self.prop_conversion_factor, self.prop_conversion_factor],
                            [-self.body_length * self.prop_conversion_factor,
                             self.body_length * self.prop_conversion_factor]])

        # This matrix is used to compute the control inputs needed for a desired force vector.
        self.h = np.matrix([[self.prop_conversion_factor, self.prop_conversion_factor],
                            [-self.body_length * self.prop_conversion_factor,
                             self.body_length * self.prop_conversion_factor]])

        self.vertical_pid = PidController((2.0, 0.01, 1.0), effort_bounds=(self.gravity, 5.0), integral_threshold=0.9)
        self.horizontal_pid = PidController((5.0, 0.0, 3.0), effort_bounds=(-1000.0, 1000.0))
        self.angular_pid = PidController((300.0, 0.0, 20.0), effort_bounds=(-8000.0, 8000.0))
        self.use_pid = True

    def physics_update(self, dt):
        c, s = cos(self.q[2]), sin(self.q[2])
        Rvb = np.asarray([[c, s, 0],
                          [-s, c, 0],
                          [0, 0, 1]])

        prop_speed_squared = [vi * vi for vi in self.prop_speeds]
        u = np.asarray(prop_speed_squared).reshape((2, 1))

        # Force vector resulting from control input u as seen in the body frame.
        F = np.matmul(self.H, u)

        # Defined from Body Frame
        q_b_dot_dot = np.matmul(np.linalg.inv(self.M), F)

        # Force of gravity from vehicle frame
        g_v = np.matrix([0, self.gravity, 0]).T
        g_b = np.matmul(Rvb, g_v)

        q_b_dot_dot += g_b
        # Acceleration vector as seen from vehicle frame.
        q_v_dot_dot = np.matmul(Rvb.T, q_b_dot_dot)

        self.q_dot += q_v_dot_dot * dt
        self.q += self.q_dot * dt

    def control_update(self, dt):
        target_altitude = self.target_alt
        target_x = self.target_x

        total_thrust = self.vertical_control_helper(target_altitude, dt)
        torque = self.horizontal_control_helper(target_x, total_thrust, dt)
        effort = np.asarray([total_thrust, torque]).reshape((2, 1)).astype(np.float64)
        # print(total_thrust, torque)
        prop_speeds_squared = np.matmul(np.linalg.inv(self.h), effort)
        prop_speeds_squared = np.clip(prop_speeds_squared, 0.0, None)
        self.prop_speeds = np.sqrt(prop_speeds_squared)

    def vertical_control_helper(self, target_altitude, dt):
        # Function Variables
        if self.use_pid:
            desired_vertical_velocity = self.vertical_pid.get_effort(target_altitude, self.q[1], dt)
            desired_vertical_velocity = np.clip(desired_vertical_velocity, -1.5, 1.5)
        else:
            desired_vertical_velocity = self.get_desired_velocity(target_altitude, self.q[1])

        delta_desired_velocity = desired_vertical_velocity - self.q_dot[1]
        c = cos(self.q[2])

        # Calculate the total desired thrust to achieve desired vertical velocity
        desired_thrust = self.mass * ((delta_desired_velocity / dt) - self.gravity) / c
        return desired_thrust

    def horizontal_control_helper(self, target_position_x, force, dt):

        if self.use_pid:
            desired_velocity = self.horizontal_pid.get_effort(target_position_x, self.q[0], dt)
            desired_velocity = np.clip(desired_velocity, -2.0, 2.0)
        else:
            desired_velocity = self.get_desired_velocity(target_position_x, self.q[0], 0.5)

        desired_acceleration = self.get_desired_acceleration(desired_velocity, self.q_dot[0], 0.5)
        force_ratio = -desired_acceleration * self.mass / force
        bounded_force_ratio = max(min(force_ratio, 1.0), -1.0)
        desired_theta = asin(bounded_force_ratio)
        if self.use_pid:
            desired_angular_velocity = self.angular_pid.get_effort(desired_theta, self.q[2], dt)
            desired_angular_velocity = np.clip(desired_angular_velocity, -100.0, 100.0)
        else:
            desired_angular_velocity = self.get_desired_velocity(desired_theta, self.q[2], 1.0)
        desired_angular_acceleration = self.get_desired_acceleration(desired_angular_velocity, self.q_dot[2], 1.0)
        desired_torque = desired_angular_acceleration * self.Icm
        return desired_torque

    def get_prop_speed_for_thrust(self, force):
        if force <= 0.0:
            return 0.0
        omega = sqrt(force / self.prop_conversion_factor)
        return omega

    def get_desired_acceleration(self, target, current, attraction_strength=1.0):
        delta = target - current
        acceleration = delta * attraction_strength
        return acceleration

    def get_desired_velocity(self, target, current, attraction_strength=1.0):
        delta = target - current
        velocity = delta * attraction_strength
        return velocity

    def draw(self):

        target_altitude = self.target_alt
        target_x = self.target_x
        self.draw_target(target_x, target_altitude)
        # Translate coordinates to center
        glPushMatrix()
        glTranslatef(self.q[0] / 2.0, self.q[1] / 2.0, 0)
        glRotatef(degrees(self.q[2]), 0, 0, 1)

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
        self.draw_motor(0)
        glPopMatrix()
        glPushMatrix()
        glTranslatef(-self.body_length, self.body_height, 0)
        self.draw_motor(1)
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

    def draw_motor(self, index):

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

        self.draw_thrust_vector(index)

    def draw_thrust_vector(self, index):
        prop_speed = self.prop_speeds[index]
        thrust = self.prop_conversion_factor * prop_speed * prop_speed
        vector_length = thrust / (200.0 * 2)
        vector_width = 0.01
        arrow_tip_size = 0.05
        arrow_color = [120, 40, 160]

        # Draw body of arrow
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                             ('v2f', [-vector_width / 2.0, 0.0,
                                      vector_width / 2.0, 0.0,
                                      vector_width / 2.0, vector_length,
                                      -vector_width / 2.0, vector_length
                                      ]),
                             ('c3B', arrow_color * 4)
                             )

        # Draw tip of arrow
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                             ('v2f', [-arrow_tip_size / 2.0, vector_length,
                                      arrow_tip_size / 2.0, vector_length,
                                      0.0, arrow_tip_size + vector_length,
                                      0.0, arrow_tip_size + vector_length
                                      ]),
                             ('c3B', arrow_color * 4)
                             )

    # draw target
    def draw_target(self, target_x_position, target_y_position):
        target_y_position = target_y_position / 2.0
        target_x_position = target_x_position / 2.0
        size = 0.1
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                             ('v2f', [target_x_position + size, target_y_position,
                                      target_x_position + size, target_y_position - size,
                                      target_x_position, target_y_position - size,
                                      target_x_position, target_y_position]),
                             ('c3B', [45, 29, 250] * 4)
                             )


class PidController:
    def __init__(self, gains=(0.0, 0.0, 0.0),
                 effort_bounds=(0.0, 0.0),
                 integral_threshold=0.25):
        self.k_p, self.k_i, self.k_d = gains
        self.e_prev, self.i_prev = None, 0.0
        self.min_effort, self.max_effort = effort_bounds
        self.integration_threshold = integral_threshold

    def get_effort(self, target, measurement, dt):
        if self.e_prev is None:
            self.e_prev = target - measurement

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
