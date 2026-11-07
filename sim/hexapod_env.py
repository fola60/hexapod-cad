import json
from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

REWARD_TERMS = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "feet_air_time",
    "lin_vel_z",
    "ang_vel_xy",
    "orientation",
    "torques",
    "action_rate",
    "dof_acc",
    "collision",
    "foot_slip",
    "dof_pos_limits",
)
INFO_KEYS = tuple(f"rew_{k}" for k in REWARD_TERMS) + ("rew_termination", "lin_vel_error", "yaw_error")


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def load_config(path):
    path = Path(path).resolve()
    return json.loads(path.read_text()), path.parent


class HexapodEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config, config_dir=None, randomize=None, noise=None, render_mode=None):
        if isinstance(config, (str, Path)):
            cfg, config_dir = load_config(config)
        else:
            cfg = json.loads(json.dumps(config))
        self.cfg = cfg
        self.config_dir = Path(config_dir) if config_dir is not None else Path.cwd()
        self.render_mode = render_mode
        self.randomize = cfg["domain_rand"]["enabled"] if randomize is None else bool(randomize)
        self.noise_enabled = cfg["obs"]["noise"]["enabled"] if noise is None else bool(noise)

        rc = cfg["robot"]
        self.model = mujoco.MjModel.from_xml_path(str((self.config_dir / rc["model"]).resolve()))
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.base_id = m.body(rc["base_body"]).id
        free = [j for j in range(m.njnt) if m.jnt_bodyid[j] == self.base_id and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError(f"base body {rc['base_body']} needs exactly one free joint")
        self.root_qadr = int(m.jnt_qposadr[free[0]])
        self.root_vadr = int(m.jnt_dofadr[free[0]])

        self.imu_id = m.site(rc["imu_site"]).id
        if m.site_bodyid[self.imu_id] != self.base_id:
            raise ValueError("IMU site must be attached to the base body")
        self.R_imu = quat_to_mat(m.site_quat[self.imu_id])

        self.nu = m.nu
        if any(m.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT for i in range(self.nu)):
            raise ValueError("all actuators must drive joints")
        jids = m.actuator_trnid[:, 0].astype(int)
        self.joint_ids = jids
        self.qadr = m.jnt_qposadr[jids].astype(int)
        self.vadr = m.jnt_dofadr[jids].astype(int)
        self.q_lo = m.jnt_range[jids, 0].copy()
        self.q_hi = m.jnt_range[jids, 1].copy()
        mid = 0.5 * (self.q_lo + self.q_hi)
        half = 0.5 * (self.q_hi - self.q_lo) * cfg["rewards"]["soft_dof_pos_limit"]
        self.soft_lo, self.soft_hi = mid - half, mid + half

        self.key_id = m.key(rc["default_keyframe"]).id
        self.key_qpos = m.key_qpos[self.key_id].copy()
        self.default_q = self.key_qpos[self.qadr].copy()

        self.floor_id = m.geom(rc["floor_geom"]).id
        self.foot_ids = np.array([m.geom(n).id for n in rc["foot_geoms"]], dtype=int)
        self.n_feet = len(self.foot_ids)
        root = m.body_rootid[self.base_id]
        robot_geom = np.array([m.body_rootid[m.geom_bodyid[g]] == root for g in range(m.ngeom)])
        self.geom_foot = np.full(m.ngeom, -1, dtype=int)
        self.geom_foot[self.foot_ids] = np.arange(self.n_feet)
        self.geom_base = np.array([m.geom_bodyid[g] == self.base_id for g in range(m.ngeom)])
        self.geom_nonfoot = robot_geom & (self.geom_foot < 0)

        cc = cfg["control"]
        for i in range(self.nu):
            if m.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED or m.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_AFFINE:
                raise ValueError("expected position actuators")
        if cc.get("kp") is not None:
            m.actuator_gainprm[:, 0] = cc["kp"]
            m.actuator_biasprm[:, 1] = -cc["kp"]
        if cc.get("kv") is not None:
            m.actuator_biasprm[:, 2] = -cc["kv"]
        if cc.get("torque_limit") is not None:
            m.actuator_forcelimited[:] = 1
            m.actuator_forcerange[:, 0] = -cc["torque_limit"]
            m.actuator_forcerange[:, 1] = cc["torque_limit"]
        self.nom_kp = m.actuator_gainprm[:, 0].copy()
        self.nom_kv = -m.actuator_biasprm[:, 2].copy()
        self.nom_forcerange = m.actuator_forcerange.copy()
        self.nom_friction = m.geom_friction.copy()
        self.nom_mass = float(m.body_mass[self.base_id])
        self.nom_inertia = m.body_inertia[self.base_id].copy()

        self.decimation = max(1, int(round(cc["policy_dt"] / m.opt.timestep)))
        self.dt = self.decimation * m.opt.timestep
        self.action_scale = cc["action_scale"]
        self.action_clip = cc["action_clip"]
        self.max_steps = int(round(cfg["episode"]["max_time_s"] / self.dt))
        g = np.asarray(m.opt.gravity, dtype=np.float64)
        self.gravity_dir = g / np.linalg.norm(g)

        sc = cfg["obs"]["scales"]
        self.s_ang = sc["ang_vel"]
        self.s_cmd = np.asarray(sc["cmd"], dtype=np.float64)
        self.s_dvel = sc["dof_vel"]
        self.s_lin = sc["lin_vel"]
        self.actor_obs_dim = 9 + 3 * self.nu
        self.priv_obs_dim = 3 + 1 + self.n_feet + 1
        self.obs_dim = self.actor_obs_dim + self.priv_obs_dim
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.nu,), np.float32)

        self.weights = {k: float(cfg["rewards"]["weights"].get(k, 0.0)) for k in REWARD_TERMS}
        self.fixed_command = None
        self.command = np.zeros(3)
        self._renderer = None

        mujoco.mj_resetDataKeyframe(m, self.data, self.key_id)
        mujoco.mj_forward(m, self.data)
        self.floor_z = float(self.data.geom_xpos[self.floor_id][2])
        self._clear_buffers(0)

    def set_command(self, vx, vy, yaw):
        self.fixed_command = np.array([vx, vy, yaw], dtype=np.float64)
        self.command = self.fixed_command.copy()

    def clear_command(self):
        self.fixed_command = None

    def _clear_buffers(self, delay):
        self.step_count = 0
        self.last_action = np.zeros(self.nu)
        self.action_buf = deque([np.zeros(self.nu) for _ in range(delay + 1)], maxlen=delay + 1)
        self.last_dof_vel = np.zeros(self.nu)
        self.feet_air_time = np.zeros(self.n_feet)
        self.last_contacts = np.zeros(self.n_feet, dtype=bool)
        self.foot_contact = np.zeros(self.n_feet, dtype=bool)
        self.next_push = np.inf

    def _sample_command(self):
        cc = self.cfg["commands"]
        rng = self.np_random
        if self.fixed_command is not None:
            self.command = self.fixed_command.copy()
        else:
            c = np.array([rng.uniform(*cc["vx"]), rng.uniform(*cc["vy"]), rng.uniform(*cc["yaw"])])
            if rng.random() < cc["zero_prob"]:
                c[:] = 0.0
            if np.linalg.norm(c[:2]) < cc["min_xy"]:
                c[:2] = 0.0
            self.command = c
        self.next_resample = self.step_count + int(round(rng.uniform(*cc["resample_s"]) / self.dt))

    def _apply_domain_rand(self):
        m, dr, rng = self.model, self.cfg["domain_rand"], self.np_random
        m.geom_friction[:] = self.nom_friction
        m.body_mass[self.base_id] = self.nom_mass
        m.body_inertia[self.base_id] = self.nom_inertia
        m.actuator_gainprm[:, 0] = self.nom_kp
        m.actuator_biasprm[:, 1] = -self.nom_kp
        m.actuator_biasprm[:, 2] = -self.nom_kv
        m.actuator_forcerange[:] = self.nom_forcerange
        if not self.randomize:
            return 0
        f = rng.uniform(*dr["friction"])
        m.geom_friction[self.floor_id, 0] = f
        m.geom_friction[self.foot_ids, 0] = f
        s = rng.uniform(*dr["base_mass_scale"])
        m.body_mass[self.base_id] = self.nom_mass * s + rng.uniform(*dr["payload_kg"])
        m.body_inertia[self.base_id] = self.nom_inertia * s
        m.actuator_forcerange[:] = self.nom_forcerange * rng.uniform(*dr["torque_limit_scale"], size=(self.nu, 1))
        kp = self.nom_kp * rng.uniform(*dr["kp_scale"], size=self.nu)
        kv = self.nom_kv * rng.uniform(*dr["kv_scale"], size=self.nu)
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 1] = -kp
        m.actuator_biasprm[:, 2] = -kv
        lo, hi = dr["action_delay_steps"]
        return int(rng.integers(lo, hi + 1))

    def _schedule_push(self):
        self.next_push = self.step_count + int(round(self.np_random.uniform(*self.cfg["domain_rand"]["push_interval_s"]) / self.dt))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        m, d, rng = self.model, self.data, self.np_random
        delay = self._apply_domain_rand()
        mujoco.mj_resetDataKeyframe(m, d, self.key_id)
        n = self.cfg["episode"]["reset_joint_noise"]
        d.qpos[self.qadr] = np.clip(self.default_q + rng.uniform(-n, n, self.nu), self.q_lo, self.q_hi)
        if self.cfg["episode"]["random_yaw"]:
            yaw = rng.uniform(-np.pi, np.pi)
            qy = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, qy, self.key_qpos[self.root_qadr + 3:self.root_qadr + 7])
            d.qpos[self.root_qadr + 3:self.root_qadr + 7] = q
        d.ctrl[:] = np.clip(self.default_q, self.q_lo, self.q_hi)
        mujoco.mj_forward(m, d)
        self._clear_buffers(delay)
        if self.randomize:
            self._schedule_push()
        self._sample_command()
        self._update_state()
        info = {f"rew_{k}": 0.0 for k in REWARD_TERMS}
        info.update(rew_termination=0.0, lin_vel_error=0.0, yaw_error=0.0)
        return self._obs(), info

    def _update_state(self):
        d = self.data
        q = d.qpos[self.root_qadr + 3:self.root_qadr + 7]
        R = quat_to_mat(q)
        self.base_lin_vel = R.T @ d.qvel[self.root_vadr:self.root_vadr + 3]
        self.base_ang_vel = d.qvel[self.root_vadr + 3:self.root_vadr + 6].copy()
        self.proj_gravity = R.T @ self.gravity_dir
        self.base_height = float(d.qpos[self.root_qadr + 2]) - self.floor_z
        self.dof_pos = d.qpos[self.qadr].copy()
        self.dof_vel = d.qvel[self.vadr].copy()

        self.foot_contact[:] = False
        self.base_contact = False
        touching = np.zeros(self.model.ngeom, dtype=bool)
        if d.ncon:
            pairs = d.contact.geom[:d.ncon]
            for g1, g2 in pairs:
                if g1 == self.floor_id:
                    other = g2
                elif g2 == self.floor_id:
                    other = g1
                else:
                    continue
                touching[other] = True
        feet = self.geom_foot[touching]
        self.foot_contact[feet[feet >= 0]] = True
        self.base_contact = bool(np.any(touching & self.geom_base))
        self.n_collisions = int(np.sum(touching & self.geom_nonfoot))

    def _foot_speed_xy2(self):
        out = np.zeros(self.n_feet)
        res = np.zeros(6)
        for k, g in enumerate(self.foot_ids):
            if self.foot_contact[k]:
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, int(g), res, 0)
                out[k] = res[3] ** 2 + res[4] ** 2
        return out

    def _obs(self):
        nc = self.cfg["obs"]["noise"]
        rng = self.np_random
        ang = self.R_imu.T @ self.base_ang_vel
        grav = self.R_imu.T @ self.proj_gravity
        dpos = self.dof_pos - self.default_q
        dvel = self.dof_vel.copy()
        if self.noise_enabled:
            ang = ang + rng.uniform(-nc["ang_vel"], nc["ang_vel"], 3)
            grav = grav + rng.uniform(-nc["gravity"], nc["gravity"], 3)
            dpos = dpos + rng.uniform(-nc["dof_pos"], nc["dof_pos"], self.nu)
            dvel = dvel + rng.uniform(-nc["dof_vel"], nc["dof_vel"], self.nu)
        actor = np.concatenate([
            ang * self.s_ang,
            grav,
            self.command * self.s_cmd,
            dpos,
            dvel * self.s_dvel,
            self.last_action,
        ])
        priv = np.concatenate([
            self.base_lin_vel * self.s_lin,
            [self.base_height],
            self.foot_contact.astype(np.float64),
            [self.model.geom_friction[self.floor_id, 0]],
        ])
        return np.nan_to_num(np.concatenate([actor, priv]), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _rewards(self, action):
        rc = self.cfg["rewards"]
        sigma = rc["tracking_sigma"]
        sigma_yaw = rc.get("tracking_sigma_yaw", sigma)
        cmd = self.command
        lin_err2 = float(np.sum((cmd[:2] - self.base_lin_vel[:2]) ** 2))
        yaw_err = float(cmd[2] - self.base_ang_vel[2])

        contact = self.foot_contact
        contact_filt = contact | self.last_contacts
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += self.dt
        air = float(np.sum((self.feet_air_time - rc["feet_air_time_target"]) * first_contact))
        if np.linalg.norm(cmd[:2]) <= rc["feet_air_time_min_cmd"]:
            air = 0.0
        self.feet_air_time *= ~contact_filt
        self.last_contacts = contact.copy()

        q = self.dof_pos
        raw = {
            "tracking_lin_vel": np.exp(-lin_err2 / sigma),
            "tracking_ang_vel": np.exp(-yaw_err ** 2 / sigma_yaw),
            "feet_air_time": air,
            "lin_vel_z": self.base_lin_vel[2] ** 2,
            "ang_vel_xy": float(np.sum(self.base_ang_vel[:2] ** 2)),
            "orientation": float(np.sum(self.proj_gravity[:2] ** 2)),
            "torques": float(np.sum(self.data.actuator_force ** 2)),
            "action_rate": float(np.sum((action - self.last_action) ** 2)),
            "dof_acc": float(np.sum(((self.dof_vel - self.last_dof_vel) / self.dt) ** 2)),
            "collision": float(self.n_collisions),
            "foot_slip": float(np.sum(self._foot_speed_xy2())),
            "dof_pos_limits": float(np.sum(np.clip(self.soft_lo - q, 0.0, None) + np.clip(q - self.soft_hi, 0.0, None))),
        }
        terms = {k: float(raw[k]) * self.weights[k] * self.dt for k in REWARD_TERMS}
        return terms, np.sqrt(lin_err2), abs(yaw_err)

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -self.action_clip, self.action_clip)
        self.action_buf.append(a)
        applied = self.action_buf[0]
        d.ctrl[:] = np.clip(self.default_q + self.action_scale * applied, self.q_lo, self.q_hi)

        if self.randomize and self.step_count >= self.next_push:
            v = self.cfg["domain_rand"]["push_vel_xy"]
            d.qvel[self.root_vadr:self.root_vadr + 2] += self.np_random.uniform(-v, v, 2)
            self._schedule_push()

        for _ in range(self.decimation):
            mujoco.mj_step(m, d)
        self.step_count += 1
        self._update_state()

        terms, lin_err, yaw_err = self._rewards(a)
        reward = sum(terms.values())
        if not np.isfinite(reward):
            terms = {k: 0.0 for k in terms}
            reward = 0.0
        if self.cfg["rewards"]["only_positive_rewards"]:
            reward = max(reward, 0.0)

        cos_tilt = -float(np.dot(self.proj_gravity, np.array([0.0, 0.0, 1.0])))
        fell = (
            self.base_contact
            or cos_tilt < np.cos(np.deg2rad(self.cfg["episode"]["fall_tilt_deg"]))
            or self.base_height < self.cfg["episode"]["min_base_height"]
            or not np.all(np.isfinite(d.qpos))
        )
        term_rew = self.cfg["rewards"]["termination_penalty"] if fell else 0.0
        reward += term_rew
        truncated = (not fell) and self.step_count >= self.max_steps

        if self.step_count >= self.next_resample:
            self._sample_command()

        self.last_action = a
        self.last_dof_vel = self.dof_vel.copy()
        obs = self._obs()
        info = {f"rew_{k}": v for k, v in terms.items()}
        info.update(rew_termination=float(term_rew), lin_vel_error=float(lin_err), yaw_error=float(yaw_err))
        return obs, float(reward), bool(fell), bool(truncated), info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
