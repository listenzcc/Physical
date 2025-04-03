import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation  # Import the correct module for animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Polygon


class VehicleSimulation:
    def __init__(self):
        # 初始化参数
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.set_xlim(-50, 150)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        self.ax.grid(True)

        # 道路参数
        self.lane_width = 3.75  # 标准车道宽度(米)
        self.road_length = 200

        # 车辆参数
        self.vehicle_length = 4.8  # 车长(米)
        self.vehicle_width = 1.8   # 车宽(米)
        self.speed = 25  # 初始速度 m/s (约90km/h)
        self.yaw_angle = 0  # 车辆偏航角
        self.position = np.array([0.0, 0.0])  # 车辆位置

        # 碰撞参数
        self.collision_time = 2.5  # 碰撞发生时间(秒)
        self.collision_detected = False

        # 创建道路元素
        self.create_road()

        # 创建车辆和护栏
        self.vehicle = Rectangle((0, 0), self.vehicle_length, self.vehicle_width,
                                 angle=0, fill=True, color='blue', alpha=0.8)
        self.ax.add_patch(self.vehicle)

        # 护栏位置 (道路中心)
        self.barrier = Rectangle((50, -0.5), 30, 1, fill=True, color='gray')
        self.ax.add_patch(self.barrier)

        # 添加施工区域标志
        construction = Rectangle((40, -self.lane_width), 20, 2*self.lane_width,
                                 fill=True, color='orange', alpha=0.3)
        self.ax.add_patch(construction)

        # 文本信息
        self.time_text = self.ax.text(
            0.02, 0.95, '', transform=self.ax.transAxes)
        self.status_text = self.ax.text(
            0.02, 0.80, '', transform=self.ax.transAxes)

        # 模拟数据
        self.time = 0
        self.dt = 0.05  # 时间步长(秒)

        # 方向盘和制动数据 (时间, 方向盘角度, 制动开度)
        self.control_data = [
            (0.0, 0.0, 0.0),     # 初始状态
            (1.0, -22.0625, 31),  # 22:44:25 左转22度，制动31%
            (2, 1.0, 38),   # 22:44:26 右转1度(从-22到-21)，制动38%
            (3.0, 1.0, 38)     # 保持
        ]

    def create_road(self):
        """创建道路可视化"""
        # 主车道
        self.ax.add_patch(Rectangle((-50, -self.lane_width), self.road_length,
                                    2*self.lane_width, fill=True, color='lightgray'))

        # 对向车道
        self.ax.add_patch(Rectangle((-50, -3*self.lane_width), self.road_length,
                                    2*self.lane_width, fill=True, color='lightgray'))

        # 车道线
        for y in [-self.lane_width, 0, self.lane_width, -2*self.lane_width, -3*self.lane_width]:
            self.ax.axhline(y, color='white' if y !=
                            0 else 'yellow', linestyle='-', linewidth=1)

        # 施工区域导向线 (临时黄色线)
        self.ax.plot([40, 60], [0, -2*self.lane_width], 'y--', linewidth=2)

    def get_controls(self, t):
        """根据时间获取当前的控制输入"""
        # 找到当前时间所在区间
        for i in range(len(self.control_data)-1):
            t1, steer1, brake1 = self.control_data[i]
            t2, steer2, brake2 = self.control_data[i+1]
            steer1 /= 15
            steer2 /= 15
            if t1 <= t < t2:
                # 线性插值
                alpha = (t - t1) / (t2 - t1)
                steer = steer1 + alpha * (steer2 - steer1)
                brake = brake1 + alpha * (brake2 - brake1)
                return steer, brake

        return self.control_data[-1][1], self.control_data[-1][2]

    def update(self, frame):
        """更新模拟状态"""
        self.time += self.dt

        # 获取当前控制输入
        steer_angle, brake_percent = self.get_controls(self.time)

        # 计算车辆动力学 (简化模型)
        steer_rad = np.deg2rad(steer_angle)

        # 制动减速度 (简化模型)
        deceleration = 0.5 * (brake_percent / 100) * 9.8  # 最大0.5g减速度

        # 更新速度
        self.speed = max(0, self.speed - deceleration * self.dt)

        # 更新偏航角 (简化自行车模型)
        wheelbase = 2.8  # 轴距(米)
        if self.speed > 0.1:
            self.yaw_angle += self.speed * \
                np.tan(steer_rad) / wheelbase * self.dt

        # 更新位置
        self.position[0] += self.speed * np.cos(self.yaw_angle) * self.dt
        self.position[1] += self.speed * np.sin(self.yaw_angle) * self.dt

        # 更新车辆位置和角度
        self.vehicle.set_xy((self.position[0] - self.vehicle_length/2,
                            self.position[1] - self.vehicle_width/2))
        self.vehicle.angle = np.rad2deg(self.yaw_angle)

        # 检测碰撞
        if True:  # not self.collision_detected and self.time >= self.collision_time:
            # 检查车辆右前角是否与护栏碰撞
            front_right_x = self.position[0] + self.vehicle_length/2 * np.cos(
                self.yaw_angle) - self.vehicle_width/2 * np.sin(self.yaw_angle)
            front_right_y = self.position[1] + self.vehicle_length/2 * np.sin(
                self.yaw_angle) + self.vehicle_width/2 * np.cos(self.yaw_angle)

            barrier_x_min, barrier_x_max = 50, 80
            barrier_y_min, barrier_y_max = -0.5, 0.5
            print(front_right_x, front_right_y)
            if front_right_x > 60:
                self.vehicle.set_color('red')

                # if (barrier_x_min <= front_right_x <= barrier_x_max and
                #         barrier_y_min <= front_right_y <= barrier_y_max):
                self.collision_detected = True
                self.vehicle.set_color('red')
                self.vehicle.set_alpha(0.6)

                # 添加碰撞碎片效果
                debris_x = front_right_x + np.random.normal(0, 0.5, 10)
                debris_y = front_right_y + np.random.normal(0, 0.5, 10)
                self.ax.scatter(debris_x, debris_y,
                                c='#333333', s=20, alpha=0.7)

        # 更新文本信息
        # self.time_text.set_text(f'Time: {self.time:.2f}s')
        status = (f'Time: {self.time:.2f}s\n'
                  f'Speed: {self.speed*3.6:.1f} km/h\n'
                  f'Steering: {steer_angle:.1f} deg\n'
                  f'Brake: {brake_percent:.0f}%')
        if self.collision_detected:
            status += '\nCOLLISION DETECTED!'
        self.status_text.set_text(status)

        # 调整视图跟随车辆
        self.ax.set_xlim(self.position[0] - 50, self.position[0] + 50)

        return self.vehicle, self.time_text, self.status_text

    def run_simulation(self):
        """运行模拟动画并保存为MP4文件"""
        total_frames = int(3.0 / self.dt)  # Calculate frames for 3.0 seconds
        anim = FuncAnimation(self.fig, self.update, frames=total_frames,
                             interval=self.dt * 1000, blit=False)

        # Save animation as MP4
        writer = matplotlib.animation.FFMpegWriter(
            fps=int(1/self.dt), metadata={'artist': 'VehicleSimulation'})
        anim.save('simulation.mp4', writer=writer)

        print("Simulation saved as 'simulation.mp4'")


# 运行模拟
sim = VehicleSimulation()
sim.run_simulation()
