'''
ADOBE CONFIDENTIAL
Copyright 2026 Adobe
All Rights Reserved.
NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def create_sphere(center, radius=0.08, resolution=20):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]

    return x, y, z


def create_equator_ring(center, radius, resolution=100):
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    z = np.zeros_like(theta) + center[2]
    return x, y, z


def create_equator_band(
    center,
    radius,
    height=0.03,
    thickness=0.01,
    resolution_theta=140,
    resolution_z=2,
    resolution_r=2,
):
    theta = np.linspace(0, 2 * np.pi, resolution_theta)
    z = np.linspace(center[2] - height / 2, center[2] + height / 2, resolution_z)

    inner_r = max(1e-6, radius - thickness / 2)
    outer_r = max(inner_r + 1e-6, radius + thickness / 2)

    TH, ZZ = np.meshgrid(theta, z)
    Xo = outer_r * np.cos(TH) + center[0]
    Yo = outer_r * np.sin(TH) + center[1]
    Zo = ZZ

    Xi = inner_r * np.cos(TH) + center[0]
    Yi = inner_r * np.sin(TH) + center[1]
    Zi = ZZ

    rr = np.linspace(inner_r, outer_r, resolution_r)
    TH2, RR = np.meshgrid(theta, rr)
    Xt = RR * np.cos(TH2) + center[0]
    Yt = RR * np.sin(TH2) + center[1]
    Zt = np.full_like(Xt, center[2] + height / 2)

    Xb = Xt.copy()
    Yb = Yt.copy()
    Zb = np.full_like(Xb, center[2] - height / 2)

    return (Xo, Yo, Zo), (Xi, Yi, Zi), (Xt, Yt, Zt), (Xb, Yb, Zb)


def create_meridian_ring(center, radius, resolution=100):
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = radius * np.cos(theta) + center[0]
    y = np.zeros_like(theta) + center[1]
    z = radius * np.sin(theta) + center[2]
    return x, y, z


def generate_trajectory(traj_type, num, step):
    t = np.linspace(0, 1, num)

    if traj_type == "forward":
        x = np.zeros(num)
        y = t * step * num
        z = np.zeros(num)

    elif traj_type == "backward":
        x = np.zeros(num)
        y = -t * step * num
        z = np.zeros(num)

    elif traj_type == "left":
        x = -t * step * num
        y = np.zeros(num)
        z = np.zeros(num)

    elif traj_type == "right":
        x = t * step * num
        y = np.zeros(num)
        z = np.zeros(num)

    elif traj_type == "s_curve":
        x = np.sin(2 * np.pi * t) * step * num * 0.5
        y = t * step * num
        z = np.zeros(num)

    elif traj_type == "loop":
        theta = 2 * np.pi * t
        r = step * num / 3
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros(num)

    else:
        raise ValueError(traj_type)

    return np.stack([x, y, z], axis=1)


class PanoramaTrajectoryVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(projection="3d")

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    def set_equal_axis_from_bounds(self, xmin, xmax, ymin, ymax, zmin, zmax, margin=0.1):
        x_mid = 0.5 * (xmin + xmax)
        y_mid = 0.5 * (ymin + ymax)
        z_mid = 0.5 * (zmin + zmax)

        max_range = max(
            xmax - xmin,
            ymax - ymin,
            zmax - zmin
        ) * (1 + margin) / 2

        self.ax.set_xlim(x_mid - max_range, x_mid + max_range)
        self.ax.set_ylim(y_mid - max_range, y_mid + max_range)
        self.ax.set_zlim(z_mid - max_range, z_mid + max_range)
        self.ax.set_box_aspect((1, 1, 1))

    def add_sphere_camera(
        self,
        center,
        color,
        radius,
        ring_offset=0.02,
        draw_equator=True,
        draw_meridian=True,
        equator_height=0.03,
        equator_thickness=0.01,
    ):
        x, y, z = create_sphere(center, radius)
        self.ax.plot_surface(
            x, y, z,
            color=color,
            linewidth=0,
            alpha=0.45,
            shade=True
        )

        ring_radius = radius + ring_offset

        if draw_equator:
            band_surfaces = create_equator_band(
                center=center,
                radius=ring_radius,
                height=equator_height,
                thickness=equator_thickness,
            )
            for X, Y, Z in band_surfaces:
                self.ax.plot_surface(
                    X, Y, Z,
                    color=color,
                    linewidth=0,
                    alpha=0.28,
                    shade=True,
                )

        if draw_meridian:
            mx, my, mz = create_meridian_ring(center, ring_radius)
            self.ax.plot(mx, my, mz, color=color, linewidth=0.8, alpha=0.75)

    def add_colorbar(self, num):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=num)
        self.fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=self.ax,
            label="Camera Index"
        )

    def save_and_show(self, title, save_path, dpi=300):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        print(f"Saved to {save_path}")
        plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_type", default="s_curve",
                        choices=["forward", "backward", "left", "right", "s_curve", "loop"])
    parser.add_argument("--num_cameras", type=int, default=40)
    parser.add_argument("--step", type=float, default=0.1)

    parser.add_argument("--camera_radius", type=float, default=0.12)
    parser.add_argument("--ring_offset", type=float, default=0.02)

    parser.add_argument("--equator_height", type=float, default=0.03,
                        help="height of the equator band along z (only when --draw_equator)")
    parser.add_argument("--equator_thickness", type=float, default=0.01,
                        help="radial thickness of the equator band (only when --draw_equator)")

    parser.add_argument("--x_offset", type=float, default=0.0)
    parser.add_argument("--y_offset", type=float, default=0.0)
    parser.add_argument("--z_offset", type=float, default=-0.3)

    parser.add_argument("--canvas_margin", type=float, default=0.05,
                        help="extra margin around content for canvas size")

    parser.add_argument("--draw_equator", action="store_true")
    parser.add_argument("--draw_meridian", action="store_true")

    parser.add_argument("--dpi", type=int, default=600,
                        help="output image DPI (higher -> higher resolution, larger file)")
    parser.add_argument("--save_path", type=str,
                        default="./panorama_camera_sphere_axes.jpg")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    centers = generate_trajectory(
        args.traj_type,
        args.num_cameras,
        args.step
    )

    centers[:, 0] += args.x_offset
    centers[:, 1] += args.y_offset
    centers[:, 2] += args.z_offset

    r = args.camera_radius + args.ring_offset
    xmin, xmax = centers[:, 0].min() - r, centers[:, 0].max() + r
    ymin, ymax = centers[:, 1].min() - r, centers[:, 1].max() + r
    zmin, zmax = centers[:, 2].min() - r, centers[:, 2].max() + r

    vis = PanoramaTrajectoryVisualizer()
    vis.set_equal_axis_from_bounds(
        xmin, xmax, ymin, ymax, zmin, zmax,
        margin=args.canvas_margin
    )

    for i, center in enumerate(centers):
        color = mpl.cm.rainbow(i / len(centers))
        vis.add_sphere_camera(
            center=center,
            color=color,
            radius=args.camera_radius,
            ring_offset=args.ring_offset,
            draw_equator=args.draw_equator,
            draw_meridian=args.draw_meridian,
            equator_height=args.equator_height,
            equator_thickness=args.equator_thickness,
        )

    vis.add_colorbar(len(centers))
    vis.save_and_show(
        title="Panorama Camera (Sphere + Equator + Meridian)",
        save_path=args.save_path,
        dpi=args.dpi
    )
