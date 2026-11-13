"""Verification Gate 3: MuJoCo Physics & Morphology Wrapper Tests (ant_morph_env/)."""

from __future__ import annotations

import numpy as np
import pytest

from ant_morph_env.wrapper import MorphologicalAntBRLWrapper


def test_gate3_50_extreme_episodes_and_morphologies() -> None:
    """Simulates 50 episodes under maximum boundary tweaks (w_i = 1000 individually & combined)
    and across 3-, 4-, and 6-legged creature variants.
    """
    rng = np.random.default_rng(2026)
    wrapper = MorphologicalAntBRLWrapper(
        max_action_capacity=12,
        fixed_action_space=True,
        default_num_legs=4,
    )
    assert "Ant-v5" in wrapper.env_id
    assert wrapper.observation_space.shape == (27,)

    num_episodes = 50
    steps_per_episode = 25

    for ep in range(num_episodes):
        tweak = np.zeros(12, dtype=np.float64)
        num_legs = 4

        if ep < 12:
            # Individual extreme tweak w_i = 1000
            tweak[ep] = 1000.0
        elif ep == 12:
            # All parameters simultaneously at maximum w = 1000
            tweak[:] = 1000.0
        elif ep == 13:
            # 3-legged creature variant
            num_legs = 3
            tweak[0] = 600.0
            tweak[3] = 1000.0
        elif ep == 14:
            # 6-legged Hexapod creature variant
            num_legs = 6
            tweak[0] = 400.0
            tweak[4] = 500.0
            tweak[11] = 800.0
        else:
            # Random extreme combinations across 3-, 4-, and 6-legged variants
            num_legs = int(rng.choice([3, 4, 6]))
            active_dims = rng.choice(12, size=4, replace=False)
            tweak[active_dims] = 1000.0
            tweak += rng.uniform(0.0, 500.0, size=12)
            tweak = np.clip(tweak, 0.0, 1000.0)

        obs, info = wrapper.reset(
            seed=100 + ep,
            options={"tweak_vector": tweak, "num_legs": num_legs},
        )
        assert np.all(np.isfinite(obs))
        assert obs.shape == (27,), f"Expected canonical obs shape (27,), got {obs.shape}"
        morph_mask = info["morphology_mask"]
        assert morph_mask.shape == (12,)

        # Verify action dimension masking for 3-, 4-, and 6-legged variants
        if num_legs == 3:
            assert np.all(morph_mask[:6] == 1.0)
            assert np.all(morph_mask[6:] == 0.0)
        elif num_legs == 4:
            assert np.all(morph_mask[:6] == 1.0)
            if tweak[3] >= 990.0:
                assert np.all(morph_mask[6:8] == 0.0)
            else:
                assert np.all(morph_mask[6:8] == 1.0)
            assert np.all(morph_mask[8:] == 0.0)
        elif num_legs == 6:
            assert np.all(morph_mask[:12] == 1.0)

        # Verify inertia matrices and masses are strictly positive and finite after mj_setConst
        assert np.all(np.isfinite(wrapper.mj_model.body_mass))
        assert np.all(wrapper.mj_model.body_mass[1:] > 0.0)
        assert np.all(np.isfinite(wrapper.mj_model.body_inertia))

        for _ in range(steps_per_episode):
            rand_action = rng.uniform(-1.0, 1.0, size=(12,)).astype(np.float32)
            obs, reward, terminated, truncated, step_info = wrapper.step(rand_action)
            assert np.all(np.isfinite(obs)), f"Non-finite obs in episode {ep}"
            assert np.isfinite(reward), f"Non-finite reward in episode {ep}"
            assert "brl_metrics" in step_info
            if terminated or truncated:
                break

    wrapper.close()


def test_standing_reward_is_unpenalized_and_win_evaluator() -> None:
    """Verifies the full env reward is returned (no stagnation penalty) and WinEvaluator still rejects zero-velocity wins."""
    from brl_core.interfaces import WinEvaluator

    wrapper = MorphologicalAntBRLWrapper(max_action_capacity=12, fixed_action_space=True)
    obs, info = wrapper.reset(seed=42)

    # Issue zero action (ant stands still)
    zero_action = np.zeros(12, dtype=np.float32)
    obs, reward, term, trunc, step_info = wrapper.step(zero_action)
    wrapper.close()

    # No stagnation penalty: standing still keeps the healthy bonus (~+1.0) from the environment
    assert np.isfinite(reward)
    assert reward > 0.5, f"Expected unpenalized env reward (healthy bonus kept), got {reward}"

    # Verify WinEvaluator: premature unhealthy termination MUST be evaluated as LOSS (0) even with high return
    we = WinEvaluator(initial_r_ref=500.0)
    win_ind, dom, ema = we.evaluate_episode(
        episode_return=900.0,
        terminated_unhealthy=True,
        forward_velocity=0.01,
    )
    assert win_ind == 0, "Unhealthy termination must evaluate to LOSS (win_indicator = 0)"
    assert dom <= 0.5, f"Dominance score for unhealthy termination should be dampened, got {dom}"

    # Verify WinEvaluator: healthy survival WITH return >= r_ref MUST be WIN (1) without velocity gate
    win_ind_run, dom_run, ema_run = we.evaluate_episode(
        episode_return=900.0,
        terminated_unhealthy=False,
        forward_velocity=0.01,
    )
    assert win_ind_run == 1, "Healthy survival meeting r_ref must evaluate to WIN (win_indicator = 1)"


def test_hexapod_all_legs_point_downward() -> None:
    """Verifies that all 6 legs of the Hexapod morphological ant point downwards towards the ground.

    Ensures that Leg 6 (middle-right) has its ankle range oriented so that throughout its valid range
    [30, 70] deg, its geom position in z is strictly below the torso (rel_z < 0), maintaining bilateral
    downward symmetry with Leg 5 (middle-left).
    """
    import mujoco
    from ant_morph_env.xml_gen import ProceduralAntXMLGenerator

    gen = ProceduralAntXMLGenerator()
    xml_str = gen.generate_xml(num_legs=6)

    m = mujoco.MjModel.from_xml_string(xml_str)
    d = mujoco.MjData(m)

    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    assert torso_id >= 0

    # Test at min, mid, and max joint angle for each leg's ankle
    for frac in [0.0, 0.5, 1.0]:
        for j in range(1, m.njnt):
            rng = m.jnt_range[j]
            d.qpos[6 + j] = rng[0] + frac * (rng[1] - rng[0])
        mujoco.mj_forward(m, d)

        torso_z = float(d.xpos[torso_id][2])
        for leg_id in range(1, 7):
            geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"ankle_{leg_id}_geom")
            assert geom_id >= 0
            rel_z = float(d.geom_xpos[geom_id][2]) - torso_z
            assert rel_z < 0.0, (
                f"Leg {leg_id} points upwards at frac={frac}: geom_z={d.geom_xpos[geom_id][2]:.4f}, "
                f"torso_z={torso_z:.4f}, rel_z={rel_z:.4f}"
            )

    # Check bilateral symmetry between Leg 5 (middle-left) and Leg 6 (middle-right) at mid-range
    for j in range(1, m.njnt):
        rng = m.jnt_range[j]
        d.qpos[6 + j] = 0.5 * (rng[0] + rng[1])
    mujoco.mj_forward(m, d)

    torso_z = float(d.xpos[torso_id][2])
    g5_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ankle_5_geom")
    g6_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ankle_6_geom")
    rel_z_5 = float(d.geom_xpos[g5_id][2]) - torso_z
    rel_z_6 = float(d.geom_xpos[g6_id][2]) - torso_z

    assert abs(rel_z_5 - rel_z_6) < 1e-4, (
        f"Leg 5 (rel_z={rel_z_5:.4f}) and Leg 6 (rel_z={rel_z_6:.4f}) are not vertically symmetric!"
    )


def test_actuator_weakening_scales_linearly() -> None:
    """Verifies w_9 and w_4 scale effective actuator torque linearly (not quadratically)."""
    wrapper = MorphologicalAntBRLWrapper(max_action_capacity=12, fixed_action_space=True)
    nom_gear = wrapper._nom_actuator_gear[:, 0].copy()
    nom_gain = wrapper._nom_actuator_gainprm[:, 0].copy()

    # w_9 = 1000 -> act_scale = 0.40 (must not become 0.16 via gainprm * gear)
    tweak_w9 = np.zeros(18, dtype=np.float32)
    tweak_w9[8] = 1000.0
    wrapper.reset(seed=42, options={"tweak_vector": tweak_w9})
    model = wrapper.mj_model
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], nom_gain, rtol=1e-6)
    np.testing.assert_allclose(model.actuator_gear[:, 0], nom_gear * 0.40, rtol=1e-6)

    # w_4 = 500 + w_9 = 1000 -> leg 4 gear = 0.40 * 0.50 = 0.20 * nom_gear
    tweak_both = np.zeros(18, dtype=np.float32)
    tweak_both[3] = 500.0
    tweak_both[8] = 1000.0
    wrapper.reset(seed=42, options={"tweak_vector": tweak_both})
    model = wrapper.mj_model
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], nom_gain, rtol=1e-6)
    np.testing.assert_allclose(model.actuator_gear[:6, 0], nom_gear[:6] * 0.40, rtol=1e-6)
    np.testing.assert_allclose(model.actuator_gear[6:8, 0], nom_gear[6:8] * 0.20, rtol=1e-6)
    wrapper.close()


def test_logarithmic_smoothness_penalty_decay() -> None:
    """Verifies logarithmic decay of action smoothness penalty from start coef down to exact 0.0 at end_step."""
    w_gentle = MorphologicalAntBRLWrapper(
        max_action_capacity=12,
        fixed_action_space=True,
        action_smoothness_coef=0.20,
        action_smoothness_end_step=500000,
        action_smoothness_log_shape=1.0,
    )
    w_steep = MorphologicalAntBRLWrapper(
        max_action_capacity=12,
        fixed_action_space=True,
        action_smoothness_coef=0.20,
        action_smoothness_end_step=500000,
        action_smoothness_log_shape=50.0,
    )
    try:
        # 1. Start penalty at step 0
        assert w_gentle.get_current_smoothness_coef() == pytest.approx(0.20, abs=1e-9)
        assert w_steep.get_current_smoothness_coef() == pytest.approx(0.20, abs=1e-9)

        # 2. Logarithmic curve & shape control at midpoint (250k): steeper shape drops lower early
        w_gentle.set_total_env_steps(250000)
        w_steep.set_total_env_steps(250000)
        mid_gentle = w_gentle.get_current_smoothness_coef()
        mid_steep = w_steep.get_current_smoothness_coef()
        assert 0.0 < mid_steep < mid_gentle < 0.10

        # 3. Exact 0.0 at and after end_step
        w_steep.set_total_env_steps(500000)
        assert w_steep.get_current_smoothness_coef() == 0.0
        w_steep.set_total_env_steps(750000)
        assert w_steep.get_current_smoothness_coef() == 0.0

        # 4. Verify step() telemetry and zero penalty beyond end_step vs non-zero penalty at step 0
        w_steep.reset(seed=42, options={"total_env_steps": 500000})
        a_pos = np.ones(12, dtype=np.float32) * 0.8
        a_neg = -np.ones(12, dtype=np.float32) * 0.8
        w_steep.step(a_pos)
        _, _, _, _, info_end = w_steep.step(a_neg)
        assert info_end["action_smoothness_coef"] == 0.0
        assert w_steep.total_env_steps == 500002

        w_steep.reset(seed=42, options={"total_env_steps": 0})
        w_steep.step(a_pos)
        _, _, _, _, info_start = w_steep.step(a_neg)
        assert 0.19 < info_start["action_smoothness_coef"] <= 0.20
    finally:
        w_gentle.close()
        w_steep.close()


def test_front_back_left_right_leg_scaling() -> None:
    """Verifies w[0] (s_front) scales +x legs (1, 4), w[1] (s_back) scales -x legs (2, 3), and w[2] scales +y/-y."""
    wrapper = MorphologicalAntBRLWrapper(max_action_capacity=12, fixed_action_space=True)
    try:
        # 1. Front scaling w[0] = 800 -> Legs 1 (+x,+y) and 4 (+x,-y) elongate; Legs 2, 3 stay nominal
        t_front = np.zeros(18, dtype=np.float32)
        t_front[0] = 800.0
        _, info_f = wrapper.reset(seed=42, options={"tweak_vector": t_front})
        kin_f = info_f["true_limb_kinematics"]
        # Anchor x/y norms: legs 1 & 4 should be > 0.5, legs 2 & 3 should be ~0.283 (sqrt(0.2^2 + 0.2^2))
        norms_f = np.linalg.norm(kin_f[:, :2], axis=1)
        assert norms_f[0] > 0.50 and norms_f[3] > 0.50
        assert norms_f[1] < 0.35 and norms_f[2] < 0.35

        # 2. Back scaling w[1] = 800 -> Legs 2 (-x,+y) and 3 (-x,-y) shrink; Legs 1, 4 stay nominal
        t_back = np.zeros(18, dtype=np.float32)
        t_back[1] = 800.0
        _, info_b = wrapper.reset(seed=42, options={"tweak_vector": t_back})
        norms_b = np.linalg.norm(info_b["true_limb_kinematics"][:, :2], axis=1)
        assert norms_b[1] < 0.20 and norms_b[2] < 0.20
        assert norms_b[0] > 0.25 and norms_b[3] > 0.25

        # 3. Left/right scaling w[2] = 800 -> +y legs (1, 2) elongate; -y legs (3, 4) shrink
        t_lr = np.zeros(18, dtype=np.float32)
        t_lr[2] = 800.0
        _, info_lr = wrapper.reset(seed=42, options={"tweak_vector": t_lr})
        norms_lr = np.linalg.norm(info_lr["true_limb_kinematics"][:, :2], axis=1)
        assert norms_lr[0] > norms_lr[3] and norms_lr[1] > norms_lr[2]
    finally:
        wrapper.close()


