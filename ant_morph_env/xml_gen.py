"""Procedural MJCF XML Generator for 3-, 4-, and 6-Legged Morphological Ant Creatures."""

from __future__ import annotations

import math
from typing import Tuple
import numpy as np


class ProceduralAntXMLGenerator:
    """Generates dynamic MuJoCo MJCF XML strings for variable-morphology Ant creatures."""

    def __init__(self, base_leg_unit: float = 0.2) -> None:
        self.base_leg_unit = float(base_leg_unit)

    @staticmethod
    def compute_leg_scales(
        w_1: float = 0.0, w_2: float = 0.0, w_3: float = 0.0
    ) -> Tuple[float, float, float, float]:
        """Computes (s_front, s_back, s_left, s_right) according to Table 4.1."""
        w1_n = float(np.clip(w_1, 0.0, 1000.0)) / 1000.0
        w2_n = float(np.clip(w_2, 0.0, 1000.0)) / 1000.0
        w3_n = float(np.clip(w_3, 0.0, 1000.0)) / 1000.0

        s_front = 1.0 + 1.2 * w1_n
        s_back = 1.0 - 0.65 * w2_n
        s_left = 1.0 + 0.45 * w3_n
        s_right = 1.0 - 0.45 * w3_n
        return s_front, s_back, s_left, s_right

    def generate_xml(
        self,
        tweak_vector: np.ndarray | None = None,
        num_legs: int = 4,
    ) -> str:
        """Generates a complete MuJoCo MJCF XML string for a 3-, 4-, or 6-legged Ant."""
        if tweak_vector is None:
            w = np.zeros(12, dtype=np.float64)
        else:
            w = np.asarray(tweak_vector, dtype=np.float64).flatten()
            if w.shape[0] < 12:
                w = np.pad(w, (0, 12 - w.shape[0]))

        s_front, s_back, s_left, s_right = self.compute_leg_scales(w[0], w[1], w[2])
        s_mid = 0.5 * (s_front + s_back)

        # Define leg configurations:
        # (leg_idx, dx, dy, ankle_ax, ankle_ay, ankle_range, scale_factor, is_left)
        all_leg_specs = [
            # Leg 1: Front-Left (+x, +y)
            (1, 1.0, 1.0, -1.0, 1.0, "30 70", s_front * s_left, True),
            # Leg 2: Back-Left (-x, +y)
            (2, -1.0, 1.0, 1.0, 1.0, "-70 -30", s_back * s_left, True),
            # Leg 3: Back-Right (-x, -y)
            (3, -1.0, -1.0, -1.0, 1.0, "-70 -30", s_back * s_right, False),
            # Leg 4: Front-Right (+x, -y)
            (4, 1.0, -1.0, 1.0, 1.0, "30 70", s_front * s_right, False),
            # Leg 5: Middle-Left (Hexapod, +y)
            (5, 0.0, 1.25, -1.0, 0.0, "30 70", s_mid * s_left, True),
            # Leg 6: Middle-Right (Hexapod, -y)
            (6, 0.0, -1.25, 1.0, 0.0, "30 70", s_mid * s_right, False),
        ]

        if num_legs == 3:
            active_specs = all_leg_specs[:3]
        elif num_legs == 6:
            active_specs = all_leg_specs[:6]
        else:
            active_specs = all_leg_specs[:4]

        max_scale = max(max(spec[6] for spec in active_specs), 1.0)
        torso_z = min(1.35, max(0.75, 0.75 * max_scale))

        leg_bodies_xml = []
        actuators_xml = []

        u0 = self.base_leg_unit
        for leg_id, dx, dy, ax, ay, a_range, scale, is_left in active_specs:
            # Clamp combined scale to >= 0.22 so capsule lengths remain well-conditioned
            eff_scale = max(0.22, float(scale))
            p1x = u0 * dx * eff_scale
            p1y = u0 * dy * eff_scale
            p2x = 2.0 * p1x
            p2y = 2.0 * p1y

            # Asymmetric default foot friction matching Ant-v4 (1.9 left vs 0.9 right)
            foot_friction = "1.9 0.1 0.1" if is_left else "0.9 0.1 0.1"

            body_str = f"""
      <body name="leg_{leg_id}" pos="0 0 0">
        <geom name="aux_{leg_id}_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 {p1x:.4f} {p1y:.4f} 0.0" friction="{foot_friction}"/>
        <body name="aux_{leg_id}" pos="{p1x:.4f} {p1y:.4f} 0">
          <joint name="hip_{leg_id}" type="hinge" pos="0.0 0.0 0.0" axis="0 0 1" range="-30 30" armature="1" damping="1"/>
          <geom name="leg_{leg_id}_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 {p1x:.4f} {p1y:.4f} 0.0" friction="{foot_friction}"/>
          <body name="foot_{leg_id}" pos="{p1x:.4f} {p1y:.4f} 0">
            <joint name="ankle_{leg_id}" type="hinge" pos="0.0 0.0 0.0" axis="{ax:.1f} {ay:.1f} 0.0" range="{a_range}" armature="1" damping="1"/>
            <geom name="ankle_{leg_id}_geom" type="capsule" size="0.08" fromto="0.0 0.0 0.0 {p2x:.4f} {p2y:.4f} 0.0" friction="{foot_friction}"/>
          </body>
        </body>
      </body>"""
            leg_bodies_xml.append(body_str)

            actuators_xml.append(
                f'    <motor name="hip_{leg_id}_motor" joint="hip_{leg_id}" ctrllimited="true" ctrlrange="-1.0 1.0" gear="150"/>'
            )
            actuators_xml.append(
                f'    <motor name="ankle_{leg_id}_motor" joint="ankle_{leg_id}" ctrllimited="true" ctrlrange="-1.0 1.0" gear="150"/>'
            )

        legs_joined = "\n".join(leg_bodies_xml)
        actuators_joined = "\n".join(actuators_xml)

        xml_str = f"""<mujoco model="morphological_ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <custom>
    <numeric data="0.0 0.0 {torso_z:.4f} 1.0 0.0 0.0 0.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1.0 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
    <hfield name="terrain" nrow="32" ncol="32" size="12 12 0.18 0.1"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <geom conaffinity="1" condim="3" name="terrain_hfield" type="hfield" hfield="terrain" pos="0 0 -0.001" rgba="0.7 0.8 0.7 1"/>
    <body name="torso" pos="0 0 {torso_z:.4f}">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
{legs_joined}
    </body>
  </worldbody>
  <actuator>
{actuators_joined}
  </actuator>
</mujoco>
"""
        return xml_str
