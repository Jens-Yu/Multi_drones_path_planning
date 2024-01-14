import matplotlib.pyplot as plt
import numpy as np
import random
import math
from matplotlib.animation import FuncAnimation

class Node:
    """RRT节点类"""
    def __init__(self, point):
        self.point = point  # 节点坐标
        self.parent = None  # 节点的父节点


def step_from_to(point1, point2, step_size):
    """从点1向点2扩展固定步长"""
    if distance(point1, point2) < step_size:
        return point2
    else:
        theta = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
        return point1[0] + step_size * np.cos(theta), point1[1] + step_size * np.sin(theta)


# 修改版的 RRT 算法，用于在有障碍物的环境中寻找从起点到终点的路径
def rrt_modified(start, goal, obstacles,num_iterations=1000):
    # 首先检查是否可以直接从起点到终点
    if is_collision_free(start, goal, obstacles):
        return [start, goal], [[start, goal]]
    nodes = [Node(start)]
    step_size = 1
    all_paths = []
    for _ in range(num_iterations):
        current_node = nodes[-1]
        if is_collision_free(current_node.point, goal, obstacles):
            goal_node = Node(goal)
            goal_node.parent = current_node
            nodes.append(goal_node)
            all_paths.append([current_node.point, goal])
            break
        if random.uniform(0, 1) < 0.5:
            random_point = goal
        else:
            random_point = (random.uniform(0, 100), random.uniform(0, 100))
        nearest_node = min(nodes, key=lambda node: distance(node.point, random_point))
        new_point = step_from_to(nearest_node.point, random_point, step_size)

        if is_collision_free(nearest_node.point, new_point, obstacles):
            new_node = Node(new_point)
            new_node.parent = nearest_node
            nodes.append(new_node)
            all_paths.append([nearest_node.point, new_point])
            if is_point_near_line(goal, nearest_node.point, new_point, 3):
                break

    path = []
    current_node = nodes[-1]
    while current_node.parent is not None:
        path.append(current_node.point)
        current_node = current_node.parent
    path.append(start)
    path.reverse()

    return path, all_paths




def distance(p1, p2=(0, 0)):
    """计算两点之间的欧几里得距离"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_point_near_line(point, line_start, line_end, radius = 4):
    """判断点是否在线段附近"""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    # 计算直线方程
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    dist_to_line = abs(A * px + B * py + C) / np.sqrt(A**2 + B**2)

    # 检查点是否在线段的范围内
    dot1 = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
    dot2 = (px - x2) * (x1 - x2) + (py - y2) * (y1 - y2)
    is_within_line_segment = dot1 >= 0 and dot2 >= 0

    return dist_to_line <= radius and is_within_line_segment


def is_collision_free(current_point, next_point, obstacles):
    """检查路径是否与障碍物相交"""
    for (ox, oy, size) in obstacles:
        if is_point_near_line((ox, oy), current_point, next_point, size):
            return False
        if distance(next_point, (ox, oy)) <= size:
            return False
    return True

# 将无人机分配给最近的取货点的函数
def assign_drones_to_pickups(drone_starts, pickup_points):
    drone_assignments = [0 for _ in range(len(drone_starts))]  # 记录每架无人机的分配次数
    pickup_assignments = {}  # 分配无人机到取货点
    drone_which_pickup = [[] for _ in range(4)]

    # 按距离原点的远近排序取货点
    sorted_pickup_points = sorted(enumerate(pickup_points), key=lambda x: -distance(x[1]))

    for i, (px, py, _) in sorted_pickup_points:
        if i in pickup_assignments:
            continue
        min_dist = float('inf')
        min_drone = None
        for j, drone_start in enumerate(drone_starts):
            if drone_assignments[j] < 2:  # 如果无人机的分配次数少于2
                dist = distance((px, py), drone_start)
                if dist < min_dist:
                    min_dist = dist
                    min_drone = j

        if min_drone is not None:
            pickup_assignments[i] = min_drone
            drone_which_pickup[min_drone].append(i)
            drone_assignments[min_drone] += 1
            # 检查并标记经过的其他取货点
            for k, (opx, opy, _) in enumerate(pickup_points):
                if k != i and k not in pickup_assignments and is_point_near_line((opx, opy), drone_starts[min_drone], (px, py)) :
                    if drone_assignments[min_drone] == 2:
                        pickup_assignments[drone_which_pickup[min_drone].pop(0)] = None
                        drone_assignments[min_drone] -= 1
                    pickup_assignments[k] = min_drone
                    drone_which_pickup[min_drone].append(k)
                    drone_assignments[min_drone] += 1
                    # 标记为已分配两个取货点
                    break
    for i, (px, py, _) in sorted_pickup_points:
        if not pickup_assignments[i]:
            min_dist = float('inf')
            min_drone = None
            for j, drone_start in enumerate(drone_starts):
                if drone_assignments[j] < 2:  # 如果无人机的分配次数少于2
                    dist = distance((px, py), drone_start)
                    if dist < min_dist:
                        min_dist = dist
                        min_drone = j

            if min_drone is not None:
                pickup_assignments[i] = min_drone
                drone_which_pickup[min_drone].append(i)
                drone_assignments[min_drone] += 1
    return drone_which_pickup

def find_nearest_delivery_point(pickup_point, delivery_points):
    """找出离指定取货点最近的收货点"""
    min_dist = float('inf')
    nearest_delivery = None
    for i, (dx, dy, _) in enumerate(delivery_points):
        dist = distance((dx, dy), pickup_point)
        if dist < min_dist:
            min_dist = dist
            nearest_delivery = i
    return nearest_delivery

# 基于无人机的取货点分配，为每架无人机确定收货点的函数
def assign_delivery_points_to_drones(drone_which_pickup, pickup_points, delivery_points):
    drone_delivery_assignments = []

    for drone_pickups in drone_which_pickup:
        nearest_delivery = find_nearest_delivery_point(pickup_points[drone_pickups[0]][:2], delivery_points)
        drone_delivery_assignments.append(nearest_delivery)
    return drone_delivery_assignments

#生成到取货点路径的函数
def generate_full_path_for_drone(drone_start, pickup_list, delivery_point, obstacles):
    if not pickup_list:  # 如果没有分配取货点
        return [], []

    full_path = []
    all_paths_animation = []  # 用于动画的路径
    current_point = drone_start

    # 生成到最后一个取货点的路径
    for pickup_idx in reversed(pickup_list):
        target = pickup_points[pickup_idx][:2]
        path_segment, paths_for_animation = rrt_modified(current_point, target, obstacles)
        if path_segment:
            full_path.extend(path_segment[1:] if full_path else path_segment)
            all_paths_animation.extend(paths_for_animation)
            current_point = path_segment[-1]

    # 生成到第一个取货点的路径
    final_target = delivery_points[delivery_point][:2]
    final_path_segment, final_paths_animation = rrt_modified(current_point, final_target, obstacles)
    if final_path_segment:
        full_path.extend(final_path_segment[1:] if full_path else final_path_segment)
        all_paths_animation.extend(final_paths_animation)

    return full_path, all_paths_animation



# 绘制地图和路径
def draw_map_with_paths(drone_paths):
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # 绘制取货点及编号
    for i, (x, y, r) in enumerate(pickup_points):
        circle = plt.Circle((x, y), r, color='red', fill=True)
        ax.add_artist(circle)
        plt.text(x, y, f'{i}', color='black')

    # 绘制收货点及编号
    for i, (x, y, r) in enumerate(delivery_points):
        circle = plt.Circle((x, y), r, color='purple', fill=True)
        ax.add_artist(circle)
        plt.text(x, y, f'{i}', color='black')

    # 绘制障碍物及编号
    for i, (x, y, r) in enumerate(obstacles):
        circle = plt.Circle((x, y), r, color='black', fill=True)
        ax.add_artist(circle)
        plt.text(x, y, f'{i}', color='white')

    # 绘制无人机起飞点及编号
    for i, (x, y) in enumerate(drone_starts):
        plt.scatter(x, y, color='green')
        plt.text(x + 1, y + 1, f'{i}', color='green')

    for path in drone_paths:
        if path:  # 如果有路径
            for i in range(len(path) - 1):
                plt.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], color='green')

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.show()

def calculate_path_length(path):
    """计算路径长度"""
    length = 0
    for i in range(len(path) - 1):
        length += distance(path[i], path[i + 1])
    return length

# 根据最长路径、最大速度和加速度计算无人机完成任务所需的最长时间的函数
def calculate_time(total_paths, max_velocity = 3, acceleration= 4/3):
    longest_path = 0
    for path in total_paths:
        length = calculate_path_length(path)
        if longest_path < length:
            longest_path = length
        # 加速到最大速度所需的距离
        distance_to_max_velocity = (max_velocity ** 2) / (2 * acceleration)

        # 检查是否能够达到最大速度
        if longest_path < 2 * distance_to_max_velocity:
            # 不能达到最大速度，仅计算加速和减速
            max_velocity_reached = math.sqrt(acceleration * longest_path)
            time_to_max_velocity = max_velocity_reached / acceleration
            total_time = 2 * time_to_max_velocity
        else:
            # 能够达到最大速度
            time_to_max_velocity = max_velocity / acceleration
            distance_at_max_velocity = longest_path - 2 * distance_to_max_velocity
            time_at_max_velocity = distance_at_max_velocity / max_velocity
            total_time = 2 * time_to_max_velocity + time_at_max_velocity

        return total_time

def calculate_path_lengths(drone_paths):
    """计算各个无人机到达其收货点的路径长度"""
    path_lengths = [calculate_path_length(path) for path in drone_paths]
    return path_lengths
# 为每架无人机规划从起点到收货点的路径并判断是否需要帮助其他无人机的函数
def get_drone_forward_paths():
    drone_forward_paths = []
    forward_paths_for_animation = []
    for i, drone_start in enumerate(drone_starts):
        pickup_list = assignments[i]
        delivery_point = delivery_assignments[i]
        drone_forward_path, forward_path_for_animation = generate_full_path_for_drone(drone_start, pickup_list, delivery_point, obstacles)
        drone_forward_paths.append(drone_forward_path)
        forward_paths_for_animation.append(forward_path_for_animation)
    # 为每架无人机计算到达收货点的路径长度
    path_lengths = calculate_path_lengths(drone_forward_paths)

    # 获取2号无人机到达其收货点的路径长度
    length_for_drone_2 = path_lengths[2]
    help_other = False
    # 对于每架其他无人机
    for i, length in enumerate(path_lengths):
        if i != 2:  # 排除2号无人机
            length_difference = length - length_for_drone_2
            if length_difference > 0:
                distance_to_drone_2_delivery = distance(delivery_points[delivery_assignments[2]][:2], delivery_points[delivery_assignments[i]][:2])
                if distance_to_drone_2_delivery < length_difference:
                    help_other = True
                    break
    if help_other:
        print("2号无人机需要协助其他无人机")
    else:
        print("2号无人机不需要协助其他无人机")
    return drone_forward_paths, forward_paths_for_animation

# 为每架无人机规划从收货点返回起点的路径的函数
def calculate_return_paths(drone_forward_paths, drone_starts, obstacles):
    drone_return_paths = []
    return_paths_for_animation = []
    for i, forward_path in enumerate(drone_forward_paths):
        if not forward_path:  # 如果没有前向路径，则跳过
            continue
        last_point = forward_path[-1]  # 获取路径的最后一个点
        start_point = drone_starts[i]  # 获取无人机的起飞点
        return_path, return_path_for_animation = rrt_modified(last_point, start_point, obstacles)  # 计算返回路径
        drone_return_paths.append(return_path)
        return_paths_for_animation.append(return_path_for_animation)
        # 打印每架无人机的返回路径
    return drone_return_paths, return_paths_for_animation

# 将前向路径和返回路径合并的函数
def combine_forward_and_return_paths(forward_paths, return_paths):
    total_paths = []
    for forward_path, return_path in zip(forward_paths, return_paths):
        if forward_path and return_path:
            # 移除返回路径中的第一个点，然后将两个路径合并
            total_path = forward_path + return_path[1:]
        elif forward_path:  # 如果只有前向路径
            total_path = forward_path
        elif return_path:  # 如果只有返回路径
            total_path = return_path
        else:  # 如果没有路径
            total_path = []

        total_paths.append(total_path)
    for i, path in enumerate(total_paths):
        print(f"无人机{i}的总路径：", path)
    return total_paths

# 获取所有无人机的前向和返回路径，用于动画展示的函数
def get_rrt_paths(forward_paths_for_animation, return_paths_for_animation):
    all_paths_for_animation = []
    for f_path, r_path in zip(forward_paths_for_animation, return_paths_for_animation):
        all_paths_for_animation.extend(f_path)
        all_paths_for_animation.extend(r_path)
    return all_paths_for_animation

# 使用 FuncAnimation 在地图上动态展示RRT的路径规划过程的函数
def animate_rrt(all_paths_for_animation):
    fig, ax = plt.subplots(figsize=(12, 12))
    for i, (x, y, r) in enumerate(pickup_points):
        circle = plt.Circle((x, y), r, color='red', fill=True)
        ax.add_artist(circle)
        plt.text(x, y, f'{i}', color='black')

        # 绘制收货点及编号
    for i, (x, y, r) in enumerate(delivery_points):
        circle = plt.Circle((x, y), r, color='purple', fill=True)
        ax.add_artist(circle)
        plt.text(x, y, f'{i}', color='black')

        # 绘制障碍物及编号
    for i, (x, y, r) in enumerate(obstacles):
        circle = plt.Circle((x, y), r, color='black', fill=True)
        ax.add_artist(circle)
        plt.text(x, y, f'{i}', color='white')

        # 绘制无人机起飞点及编号
    for i, (x, y) in enumerate(drone_starts):
        plt.scatter(x, y, color='green')
        plt.text(x + 1, y + 1, f'{i}', color='green')

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)

    # 计算动画的总帧数
    total_frames = len(all_paths_for_animation)

    #动画更新函数
    def update(frame):
        # 清除之前的路径
        ax.clear()
        for i, (x, y, r) in enumerate(pickup_points):
            circle = plt.Circle((x, y), r, color='red', fill=True)
            ax.add_artist(circle)
            plt.text(x, y, f'{i}', color='black')

            # 绘制收货点及编号
        for i, (x, y, r) in enumerate(delivery_points):
            circle = plt.Circle((x, y), r, color='purple', fill=True)
            ax.add_artist(circle)
            plt.text(x, y, f'{i}', color='black')

            # 绘制障碍物及编号
        for i, (x, y, r) in enumerate(obstacles):
            circle = plt.Circle((x, y), r, color='black', fill=True)
            ax.add_artist(circle)
            plt.text(x, y, f'{i}', color='white')

            # 绘制无人机起飞点及编号
        for i, (x, y) in enumerate(drone_starts):
            plt.scatter(x, y, color='green')
            plt.text(x + 1, y + 1, f'{i}', color='green')


        # 绘制到当前帧的 RRT 路径
        for i in range(frame):
            ax.plot([all_paths_for_animation[i][0][0], all_paths_for_animation[i][1][0]], [all_paths_for_animation[i][0][1], all_paths_for_animation[i][1][1]], linestyle='--', color='blue')


        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid(True)
    # 设置动画
    anim = FuncAnimation(fig, update, frames=range(total_frames))

    # 显示动画
    plt.show()



# 地图上的位置和半径
pickup_points = [(20, 30, 3), (50, 40, 3), (10, 80, 3), (10, 50, 3), (40, 60, 3), (30, 20, 3), (40, 90, 3)]
delivery_points = [(80, 70, 3), (70, 10, 3), (70, 80, 3), (80, 20, 3)]
obstacles = [(30, 50, 4), (40, 20, 2), (50, 70, 4), (60, 30, 3), (70, 50, 2), (80, 80, 2), (80, 10, 2), (60, 50, 4),
             (40, 10, 3), (20, 80, 5), (60, 85, 5)]
drone_starts = [(4, 4), (4, 8), (8, 4), (8, 8)]


assignments = assign_drones_to_pickups(drone_starts, pickup_points)  #分配取货点
delivery_assignments = assign_delivery_points_to_drones(assignments, pickup_points, delivery_points)  #分配收货点

print("无人机分配的取货点：", assignments)
print("无人机分配的收货点：", delivery_assignments)

# 为每架无人机规划路径
drone_forward_paths, forward_paths_for_animation = get_drone_forward_paths()
drone_return_paths, return_paths_for_animation = calculate_return_paths(drone_forward_paths, drone_starts, obstacles)

# 合并前向路径和返回路径
total_paths = combine_forward_and_return_paths(drone_forward_paths, drone_return_paths)
# 绘制总路径
draw_map_with_paths(total_paths)
# 合并去程和返程的RRT路径
all_paths_for_animation = get_rrt_paths(forward_paths_for_animation, return_paths_for_animation)
print(all_paths_for_animation)
print(calculate_time(total_paths))
animate_rrt(all_paths_for_animation)  #调用动画函数进行动画展示






