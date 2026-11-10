"""Verification Gate 3: MuJoCo Physics & Morphology Wrapper Tests (ant_morph_env/)."""

from __future__ import annotations

import numpy as np

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
