#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from collections import deque

class CartPoleLQRController(Node):
    def __init__(self):
        super().__init__('cart_pole_lqr_controller')
        
        # System parameters
        self.M = 1.0  # Mass of cart (kg)
        self.m = 1.0  # Mass of pole (kg)
        self.L = 1.0  # Length of pole (m)
        self.g = 9.81  # Gravity (m/s^2)
        
        # State space matrices
        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, (self.m * self.g) / self.M, 0],
            [0, 0, 0, 1],
            [0, 0, ((self.M + self.m) * self.g) / (self.M * self.L), 0]
        ])
        
        self.B = np.array([
            [0],
            [1/self.M],
            [0],
            [-1/(self.M * self.L)]
        ])
        
        # LQR cost matrices
        self.Q = np.diag([15.0, 15.0, 2, 10.0])  # State cost [X,Xdot,Theta,Thetadot]^T
        self.R = np.array([[0.2]])  # Control cost

        # increase X keeps cart near center but at cost of stability
        # increase Xdot Damping to keep cart from thrashing
        # increase theta fights falling over more
        # increase thetadot adds damping to reduce oscilation

        # Increase R to be a less aggressive force
        # Decrease R to be a more aggressive force

        # Compute LQR gain matrix
        self.K = self.compute_lqr_gain()
        self.get_logger().info(f'LQR Gain Matrix: {self.K}')
        
        # Initialize state estimate
        self.x = np.zeros((4, 1))
        self.state_initialized = False
        self.last_control = 0.0
        self.control_count = 0
        
        # Data storage for plotting
        self.time_steps = deque()
        self.cart_positions = deque()
        self.pole_angles = deque()
        self.control_forces = deque()
        self.start_time = None
        self.earthquake_forces = deque()
        self._warned_pre_state_earthquake = False
        
        # Create publishers and subscribers
        self.cart_cmd_pub = self.create_publisher(Float64, '/model/cart_pole/joint/cart_to_base/cmd_force', 10)
        
        if self.cart_cmd_pub:
            self.get_logger().info('Force command publisher created successfully')
        
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10) # Locks to native ROS stream
        
        self.earthquake_sub = self.create_subscription(Float64, '/earthquake_force', self.earthquake_callback, 10)
        
        # Control loop timer
        self.timer = self.create_timer(0.01, self.control_loop)

        self.MAX_SIMULATION_TIME = 120.0  # Set to desired duration
        
        self.get_logger().info('Cart-Pole LQR Controller initialized')
    
    def compute_lqr_gain(self):
        """Compute the LQR gain matrix K."""
        P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R) @ self.B.T @ P
        return K
    
    def joint_state_callback(self, msg):
        """Update state estimate from joint states."""
        try:
            cart_idx = msg.name.index('cart_to_base')
            pole_idx = msg.name.index('pole_joint')
            
            self.x = np.array([
                [msg.position[cart_idx]],
                [msg.velocity[cart_idx]],
                [msg.position[pole_idx]],
                [msg.velocity[pole_idx]]
            ])
            
            if not self.state_initialized:
                self.get_logger().info(f'Initial state: cart_pos={msg.position[cart_idx]:.3f}, cart_vel={msg.velocity[cart_idx]:.3f}, pole_angle={msg.position[pole_idx]:.3f}, pole_vel={msg.velocity[pole_idx]:.3f}')
                self.state_initialized = True
                self.start_time = self.get_clock().now().nanoseconds / 1e9
                
        except (ValueError, IndexError) as e:
            self.get_logger().warn(f'Failed to process joint states: {e}, msg={msg.name}')
    
    def earthquake_callback(self, msg):
        """Store earthquake force values."""
        if self.state_initialized:
            self.earthquake_forces.append(msg.data)
        elif not self._warned_pre_state_earthquake:
            self.get_logger().warn("Received earthquake force before state was initialized.")
            self._warned_pre_state_earthquake = True


    def print_metrics(self):
        """Prints performance metrics after simulation ends."""
        duration = self.time_steps[-1] if self.time_steps else 0.0
        max_cart_displacement = max(map(abs, self.cart_positions), default=0.0)
        max_pole_deviation = max(map(abs, self.pole_angles), default=0.0)
        avg_control_effort = np.mean(np.abs(self.control_forces)) if self.control_forces else 0.0
        stability_score = max(0, 10 - (max_cart_displacement * 2) - (max_pole_deviation / 5) - (avg_control_effort / 20))
        rms_cart_error = np.sqrt(np.mean(np.square(self.cart_positions))) if self.cart_positions else 0.0 
        # RMS takes the cart position relative to the origin and finds its average deviation with larger deviations weighted more heavily



        self.get_logger().info(f"Q values: {self.Q.diagonal()}, R values: {self.R}")
        self.get_logger().info(f"Duration of stable operation: {duration:.2f} s")
        self.get_logger().info(f"Maximum cart displacement: {max_cart_displacement:.3f} m")
        self.get_logger().info(f"Maximum pendulum angle deviation: {max_pole_deviation:.3f}°")
        self.get_logger().info(f"Average control effort: {avg_control_effort:.3f} N")
        self.get_logger().info(f"Stability score: {stability_score:.2f}/10")
        self.get_logger().info(f"RMS Cart Position Error: {rms_cart_error:.3f} m")



    def control_loop(self):
        """Compute and apply LQR control."""
        try:
            if not self.state_initialized:
                self.get_logger().warn('State not initialized yet')
                return

            u = -self.K @ self.x
            force = float(u[0])
            
            msg = Float64()
            msg.data = force
            self.cart_cmd_pub.publish(msg)
            
            self.last_control = force
            self.control_count += 1
            
            # Ensure time steps are synchronized
            current_time = self.get_clock().now().nanoseconds / 1e9 - self.start_time
            self.time_steps.append(current_time)
            self.cart_positions.append(self.x[0, 0])
            self.pole_angles.append(np.degrees(self.x[2, 0]))
            self.control_forces.append(force)

            # Ensure earthquake force logging matches other data dimensions
            if len(self.earthquake_forces) < len(self.time_steps):
                self.earthquake_forces.append(self.earthquake_forces[-1] if self.earthquake_forces else 0.0)

            # **Termination Conditions**
            if (
                abs(self.x[0, 0]) > 2.5 or 
                abs(self.x[2, 0]) > np.radians(45) or 
                current_time >= self.MAX_SIMULATION_TIME  # Stop after max time
            ):
                self.get_logger().warn(f"Simulation ended: cart_x={self.x[0, 0]:.2f}m, pole_angle={np.degrees(self.x[2, 0]):.2f}°, duration={current_time:.2f}s")
                self.print_metrics()
                self.plot_results()
                rclpy.shutdown()
                return

        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')

    def plot_results(self):
        """Generate plots for analysis."""
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.time_steps, self.cart_positions, label='Cart Position (m)', color='b')
        plt.xlabel('Time (s)')
        plt.ylabel('Cart Position (m)')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(self.time_steps, self.pole_angles, label='Pole Angle (°)', color='r')
        plt.xlabel('Time (s)')
        plt.ylabel('Pole Angle (°)')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(self.time_steps, self.earthquake_forces, label='Earthquake Force (N)', color='g')
        plt.xlabel('Time (s)')
        plt.ylabel('Earthquake Force (N)')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(self.time_steps, self.control_forces, label='Control Force (N)', color='m')  # Changed 'p' to 'm' (magenta)
        plt.xlabel('Time (s)')
        plt.ylabel('Control Force (N)')
        plt.legend()

        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    controller = CartPoleLQRController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
