#!/usr/bin/env python3
"""
Listen to /joy and print which button indices change to 1 (pressed).
Run while pressing your controller buttons to discover indices for select/start/turbo/home.
"""
import sys
import argparse

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Joy
except Exception:
    # allow syntax check on systems without ROS present
    rclpy = None
    Joy = None


class JoyWatcher(Node):
    def __init__(self, topic: str = '/joy'):
        super().__init__('joy_watcher')
        self.subscription = self.create_subscription(Joy, topic, self.cb, 10)
        self.prev_buttons = []
        self.prev_axes = []
        self.get_logger().info(f'Listening to {topic} - press controller buttons now...')

    def cb(self, msg: Joy):
        buttons = list(msg.buttons)
        axes = list(msg.axes)
        # init prev
        if not self.prev_buttons:
            self.prev_buttons = [0] * len(buttons)
        if not self.prev_axes:
            self.prev_axes = [0.0] * len(axes)
        # detect button press events
        pressed = [i for i, (p, n) in enumerate(zip(self.prev_buttons, buttons)) if p == 0 and n == 1]
        if pressed:
            self.get_logger().info(f'Buttons pressed (indices): {pressed}')
            self.get_logger().info(f'Full buttons array: {buttons}')
        # detect dpad-like axis changes (axes 6/7 common)
        for i, (p, n) in enumerate(zip(self.prev_axes, axes)):
            if p != n:
                # only log significant changes
                if abs(n - p) > 1e-3:
                    self.get_logger().info(f'Axis {i} changed: {p} -> {n}')
        self.prev_buttons = buttons[:]
        self.prev_axes = axes[:]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/joy')
    args = parser.parse_args(argv)

    if rclpy is None:
        print('rclpy not available in this environment. Run this script on a machine with ROS2 and a /joy publisher.')
        return 1

    rclpy.init()
    node = JoyWatcher(topic=args.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
