import json
import argparse
import random
import matplotlib.pyplot as plt
import taichi as ti
import math
import numpy as np
import os

random.seed(0)
np.random.seed(0)

real = ti.f32
ti.init(default_fp=real)

max_steps = 4096 *3 
vis_interval = 256
output_vis_interval = 8
steps = 2048 *3// 3
assert steps * 2 <= max_steps

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()

x = vec()
v = vec()
v_inc = vec()

head_id = 0
goal = vec()

n_objects = 0
elasticity = 0.0
ground_height = 0.1
gravity = -4.8
friction = 2.5

gradient_clip = 1
spring_omega = 10
damping = 15

n_springs = 0
n_angles = 0
angle_anchor_a = ti.field(ti.i32)
angle_anchor_b = ti.field(ti.i32)
angle_anchor_c = ti.field(ti.i32)
angle_stiffness = scalar()
spring_anchor_a = ti.field(ti.i32)
spring_anchor_b = ti.field(ti.i32)
spring_length = scalar()
spring_stiffness = scalar()
spring_actuation = scalar()
stair_heights = scalar()
stair_widths = scalar()

n_sin_waves = 10
weights1 = scalar()
bias1 = scalar()

n_hidden = 32
weights2 = scalar()
bias2 = scalar()
hidden = scalar()
center = vec()
act = scalar()

best_weights1 = scalar()
best_bias1 = scalar()
best_weights2 = scalar()
best_bias2 = scalar()


def n_input_states():
    return n_sin_waves + 4 * n_objects + 2


def allocate_fields():
    ti.root.dense(ti.i, max_steps).dense(ti.j, n_objects).place(x, v, v_inc)
    ti.root.dense(ti.i, n_springs).place(
        spring_anchor_a,
        spring_anchor_b,
        spring_length,
        spring_stiffness,
        spring_actuation,
    )
    ti.root.dense(ti.i, n_angles).place(
        angle_anchor_a, angle_anchor_b, angle_anchor_c, angle_stiffness
    )
    ti.root.dense(ti.i, options.stairs).place(stair_heights, stair_widths)
    ti.root.dense(ti.ij, (n_hidden, n_input_states())).place(weights1)
    ti.root.dense(ti.i, n_hidden).place(bias1)
    ti.root.dense(ti.ij, (n_springs, n_hidden)).place(weights2)
    ti.root.dense(ti.i, n_springs).place(bias2)
    ti.root.dense(ti.ij, (max_steps, n_hidden)).place(hidden)
    ti.root.dense(ti.ij, (max_steps, n_springs)).place(act)
    ti.root.dense(ti.i, max_steps).place(center)
    ti.root.place(loss, goal)
    ti.root.lazy_grad()

    ti.root.dense(ti.ij, (n_hidden, n_input_states())).place(best_weights1)
    ti.root.dense(ti.i, n_hidden).place(best_bias1)
    ti.root.dense(ti.ij, (n_springs, n_hidden)).place(best_weights2)
    ti.root.dense(ti.i, n_springs).place(best_bias2)



dt = 0.004
learning_rate = 25


@ti.kernel
def compute_center(t: ti.i32):
    for _ in range(1):
        c = ti.Vector([0.0, 0.0])
        for i in ti.static(range(n_objects)):
            c += x[t, i]
        center[t] = (1.0 / n_objects) * c


@ti.kernel
def nn1(t: ti.i32):
    for i in range(n_hidden):
        actuation = 0.0
        for j in ti.static(range(n_sin_waves)):
            actuation += weights1[i, j] * ti.sin(
                spring_omega * t * dt + 2 * math.pi / n_sin_waves * j
            )
        for j in ti.static(range(n_objects)):
            offset = x[t, j] - center[t]
            # use a smaller weight since there are too many of them
            actuation += weights1[i, j * 4 + n_sin_waves] * offset[0] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 1] * offset[1] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 2] * v[t, j][0] * 0.05
            actuation += weights1[i, j * 4 + n_sin_waves + 3] * v[t, j][1] * 0.05
        actuation += weights1[i, n_objects * 4 + n_sin_waves] * (
            goal[None][0] - center[t][0]
        )
        actuation += weights1[i, n_objects * 4 + n_sin_waves + 1] * (
            goal[None][1] - center[t][1]
        )
        actuation += bias1[i]
        actuation = ti.tanh(actuation)
        hidden[t, i] = actuation


@ti.kernel
def nn2(t: ti.i32):
    for i in range(n_springs):
        actuation = 0.0
        for j in ti.static(range(n_hidden)):
            actuation += weights2[i, j] * hidden[t, j]
        actuation += bias2[i]
        actuation = ti.tanh(actuation)
        act[t, i] = actuation


@ti.kernel
def apply_spring_force(t: ti.i32):
    for i in range(n_springs):
        a = spring_anchor_a[i]
        b = spring_anchor_b[i]
        pos_a = x[t, a]
        pos_b = x[t, b]
        dist = pos_a - pos_b
        length = dist.norm() + 1e-4

        target_length = spring_length[i] * (1.0 + spring_actuation[i] * act[t, i])
        impulse = dt * (length - target_length) * spring_stiffness[i] / length * dist

        ti.atomic_add(v_inc[t + 1, a], -impulse)
        ti.atomic_add(v_inc[t + 1, b], impulse)


@ti.kernel
def apply_angle_spring_force(t: ti.i32):
    # constraint between each spring with e to 90 degree
    for i in range(n_angles):
        a = angle_anchor_a[i]
        b = angle_anchor_b[i]
        c = angle_anchor_c[i]
        pos_a = x[t, a]
        pos_b = x[t, b]
        pos_c = x[t, c]
        dist_ac = pos_a - pos_c
        dist_bc = pos_b - pos_c
        angle = dist_ac.dot(dist_bc) / (dist_ac.norm() * dist_bc.norm() + 1e-4)
        angle = ti.math.acos(angle)
        angle_diff = angle - math.pi / 2
        impulse = dt * angle_diff * angle_stiffness[i] * (pos_b - pos_a).normalized()
        ti.atomic_add(v_inc[t + 1, a], impulse)
        ti.atomic_add(v_inc[t + 1, b], -impulse)


use_toi = False


@ti.kernel
def advance_toi(t: ti.i32):
    for i in range(n_objects):
        s = ti.math.exp(-dt * damping)
        old_v = s * v[t - 1, i] + dt * gravity * ti.Vector([0.0, 1.0]) + v_inc[t, i]
        old_x = x[t - 1, i]
        new_x = old_x + dt * old_v
        toi = 0.0
        new_v = old_v
        prev = 0.0
        for stair in range(options.stairs):
            stair_start = prev
            stair_end = prev + stair_widths[stair]
            if (
                stair_start < old_x[0] < stair_end
                and new_x[1] < stair_heights[stair]
                and old_v[1] < 0
                and old_x[1] >= stair_heights[stair]
            ):
                toi = -(old_x[1] - stair_heights[stair]) / old_v[1]
                new_v = ti.Vector([0.0, 0.0])
                # new_v[0] = 0.5 * old_v[0]
            prev += stair_widths[stair]
        new_x = old_x + toi * old_v + (dt - toi) * new_v
        v[t, i] = new_v
        x[t, i] = new_x


@ti.kernel
def advance_no_toi(t: ti.i32):
    for i in range(n_objects):
        s = math.exp(-dt * damping)
        old_v = s * v[t - 1, i] + dt * gravity * ti.Vector([0.0, 1.0]) + v_inc[t, i]
        old_x = x[t - 1, i]
        new_v = old_v
        # depth = old_x[1] - ground_height
        prev = 0.0
        for stair in range(options.stairs):
            x_start = prev
            x_end = prev + stair_widths[stair]
            if (
                old_x[0] < x_end
                and old_x[0] > x_start
                and old_x[1] < stair_heights[stair]
                and new_v[1] < 0
            ):
                # friction projection
                new_v[0] *= 0.5
                new_v[1] = 0
            prev += stair_widths[stair]
        new_x = old_x + dt * new_v
        v[t, i] = new_v
        x[t, i] = new_x


@ti.kernel
def compute_loss(t: ti.i32):
    loss[None] = -x[t,head_id][0]
    #loss[None] = (goal[None] - x[t, head_id]).norm()


#gui = ti.GUI("Mass Spring Robot", (512, 512), background_color=0xFFFFFF)
gui = ti.GUI(show_gui=False)


def forward(output=None, visualize=True):
    goal[None] = options.goal

    interval = vis_interval
    if output:
        interval = output_vis_interval
        os.makedirs("mass_spring/{}/".format(output), exist_ok=True)

    total_steps = steps if not output else steps * 2

    for t in range(1, total_steps):
        compute_center(t - 1)
        nn1(t - 1)
        nn2(t - 1)
        apply_spring_force(t - 1)
        apply_angle_spring_force(t - 1)
        if use_toi:
            advance_toi(t)
        else:
            advance_no_toi(t)

        if (t + 1) % interval == 0:
            # draw the stairs
            prev = 0
            for i in range(options.stairs):
                begin = (prev, options.stair_heights[i])
                end = (prev + options.stair_widths[i], options.stair_heights[i])
                gui.line(begin=begin, end=end, color=(0x0), radius=3)
                prev += options.stair_widths[i]

            # output the environment image before we draw the robot
            # only if it does not already exist. 
            if not visualize and not os.path.exists(f"env_imgs/{options.envimg_fname}"):
                gui.show(f"env_imgs/{options.envimg_fname}")
                loss[None] = 0
                compute_loss(steps - 1)
                return

            def circle(x, y, color):
                gui.circle((x, y), ti.rgb_to_hex(color), 7)

            for i in range(n_springs):

                def get_pt(x):
                    return (x[0], x[1])

                a = act[t - 1, i] * 0.5
                r = 2
                if spring_actuation[i] == 0:
                    a = 0
                    c = 0x222222
                else:
                    r = 4
                    c = ti.rgb_to_hex((0.5 + a, 0.5 - abs(a), 0.5 - a))
                gui.line(
                    begin=get_pt(x[t, spring_anchor_a[i]]),
                    end=get_pt(x[t, spring_anchor_b[i]]),
                    radius=r,
                    color=c,
                )

            for i in range(n_objects):
                color = (0.4, 0.6, 0.6)
                if i == head_id:
                    color = (0.8, 0.2, 0.3)
                circle(x[t, i][0], x[t, i][1], color)
            circle(goal[None][0], goal[None][1], (0.6, 0.2, 0.2))

            if output:
                gui.show("mass_spring/{}/{:04d}.png".format(output, t))
            else:
                gui.show()

    loss[None] = 0
    compute_loss(total_steps - 1)
    return loss[None]


@ti.kernel
def clear_states():
    for t in range(0, max_steps):
        for i in range(0, n_objects):
            v_inc[t, i] = ti.Vector([0.0, 0.0])


def clear():
    clear_states()


def setup_robot(objects, springs, angle_springs):
    global n_objects, n_springs, n_angles
    n_objects = len(objects)
    n_springs = len(springs)
    n_angles = len(angle_springs)
    allocate_fields()

    print(f"n_objects={n_objects},n_springs={n_springs},n_angles={n_angles}")
    print(
        f"stairs={options.stairs},stair_widths={options.stair_widths},stair_heights={options.stair_heights}"
    )

    for i in range(n_objects):
        x[0, i] = objects[i]

    for i in range(n_springs):
        s = springs[i]
        spring_anchor_a[i] = s[0]
        spring_anchor_b[i] = s[1]
        spring_length[i] = s[2]
        spring_stiffness[i] = s[3]
        spring_actuation[i] = s[4]

    for i in range(n_angles):
        a = angle_springs[i]
        angle_anchor_a[i] = a[0]
        angle_anchor_b[i] = a[1]
        angle_anchor_c[i] = a[2]
        angle_stiffness[i] = a[3]

    for i in range(options.stairs):
        stair_heights[i] = options.stair_heights[i]
        stair_widths[i] = options.stair_widths[i]


def optimize(toi, visualize):
    global use_toi
    use_toi = toi
    for i in range(n_hidden):
        for j in range(n_input_states()):
            weights1[i, j] = (
                np.random.randn() * math.sqrt(2 / (n_hidden + n_input_states())) * 2
            )

    for i in range(n_springs):
        for j in range(n_hidden):
            # TODO: n_springs should be n_actuators
            weights2[i, j] = (
                np.random.randn() * math.sqrt(2 / (n_hidden + n_springs)) * 3
            )

    losses = []
    # forward('initial{}'.format(robot_id), visualize=visualize)
    for iter in range(options.iters):
        clear()
        # automatically clears all gradients
        with ti.ad.Tape(loss):
            forward(visualize=visualize)

        print("Iter=", iter, "Loss=", loss[None])

        total_norm_sqr = 0
        for i in range(n_hidden):
            for j in range(n_input_states()):
                total_norm_sqr += weights1.grad[i, j] ** 2
            total_norm_sqr += bias1.grad[i] ** 2

        for i in range(n_springs):
            for j in range(n_hidden):
                total_norm_sqr += weights2.grad[i, j] ** 2
            total_norm_sqr += bias2.grad[i] ** 2

        print(total_norm_sqr)

        # scale = learning_rate * min(1.0, gradient_clip / total_norm_sqr ** 0.5)
        if (iter > 0 and min(losses) > loss[None]) or iter == 0:
            best_weights1.copy_from(weights1)
            best_bias1.copy_from(bias1)
            best_weights2.copy_from(weights2)
            best_bias2.copy_from(bias2)
            
        gradient_clip = 0.2
        scale = gradient_clip / (total_norm_sqr**0.5 + 1e-6)
        for i in range(n_hidden):
            for j in range(n_input_states()):
                weights1[i, j] -= scale * weights1.grad[i, j]
            bias1[i] -= scale * bias1.grad[i]

        for i in range(n_springs):
            for j in range(n_hidden):
                weights2[i, j] -= scale * weights2.grad[i, j]
            bias2[i] -= scale * bias2.grad[i]
        losses.append(loss[None])

    return losses


parser = argparse.ArgumentParser()
parser.add_argument("fname", type=str, help="config filename")
parser.add_argument("task", type=str, help="train/plot")
parser.add_argument("losses_fname", type=str, help="losses filename")
parser.add_argument("envimg_fname", type=str, help="filename of environment image")
parser.add_argument("-stairs", type=int, help="the number of stairs in the environment")
parser.add_argument(
    "-stair-widths",
    nargs="+",
    type=float,
    help="width of the stairs, num. arguments must match -stairs",
)
parser.add_argument(
    "-stair-heights",
    nargs="+",
    type=float,
    help="heights of the stairs, num. arguments must match -stairs",
)
parser.add_argument(
    "-goal", type=float, nargs=2, help="x,y coordinate of the goal", default=[0.9, 0.2]
)
parser.add_argument("--iters", type=int, default=50)
options = parser.parse_args()


def main():
    setup_robot(**json.load(open(options.fname)))

    if options.task == "plot":
        losses = optimize(toi=True, visualize=False)
        with open(options.losses_fname, "a+") as f:
            f.write('{"' + f"{options.fname}" + '"' + f":{losses}" + "}" + "\n")
    else:
        optimize(toi=True, visualize=True)
        clear()
        final_loss = forward("final{}".format(options.fname))
        with open(options.fname.replace(".json", ".txt"), "w") as f:
            f.write(str(final_loss))

        x_np = x.to_numpy()[:, head_id, 0]
        x_np = x_np - x_np[0]
        np.savetxt(options.fname.replace(".json", "x.txt"), x_np, delimiter=',') 

if __name__ == "__main__":
    main()
