from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from hexapod_env import REWARD_TERMS, HexapodEnv


def quat_mul(a, b):
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    return jnp.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rotate_inv(q, v):
    u = -q[1:]
    t = 2.0 * jnp.cross(u, v)
    return v + q[0] * t + jnp.cross(u, t)


class Params(NamedTuple):
    friction: jax.Array
    mass_scale: jax.Array
    payload: jax.Array
    kp: jax.Array
    kv: jax.Array
    force_scale: jax.Array
    delay: jax.Array


class State(NamedTuple):
    data: mjx.Data
    params: Params
    rng: jax.Array
    command: jax.Array
    resample_at: jax.Array
    push_at: jax.Array
    step_count: jax.Array
    last_action: jax.Array
    action_buf: jax.Array
    last_dof_vel: jax.Array
    air_time: jax.Array
    last_contacts: jax.Array


class MJXHexapodEnv:
    def __init__(self, config, config_dir=None, num_envs=4096, randomize=None, noise=None):
        cpu = HexapodEnv(config, config_dir=config_dir, randomize=False, noise=False)
        self.cpu = cpu
        self.cfg = cpu.cfg
        self.config_dir = cpu.config_dir
        self.num_envs = num_envs
        self.randomize = self.cfg["domain_rand"]["enabled"] if randomize is None else bool(randomize)
        self.noise_enabled = self.cfg["obs"]["noise"]["enabled"] if noise is None else bool(noise)

        m = cpu.model
        mc = self.cfg.get("mjx", {})
        if mc.get("iterations") is not None:
            m.opt.iterations = int(mc["iterations"])
        if mc.get("ls_iterations") is not None:
            m.opt.ls_iterations = int(mc["ls_iterations"])
        self.mx = mjx.put_model(m)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, cpu.key_id)
        mujoco.mj_forward(m, d)
        self.d0 = mjx.put_data(m, d)

        self.nu = cpu.nu
        self.n_feet = cpu.n_feet
        self.ngeom = m.ngeom
        self.obs_dim = cpu.obs_dim
        self.actor_obs_dim = cpu.actor_obs_dim
        self.dt = cpu.dt
        self.decimation = cpu.decimation
        self.max_steps = cpu.max_steps
        self.ra = cpu.root_qadr
        self.rv = cpu.root_vadr
        self.floor_id = cpu.floor_id
        self.floor_z = cpu.floor_z

        f32 = lambda x: jnp.asarray(x, dtype=jnp.float32)
        self.qadr = jnp.asarray(cpu.qadr)
        self.vadr = jnp.asarray(cpu.vadr)
        self.q_lo, self.q_hi = f32(cpu.q_lo), f32(cpu.q_hi)
        self.soft_lo, self.soft_hi = f32(cpu.soft_lo), f32(cpu.soft_hi)
        self.default_q = f32(cpu.default_q)
        self.key_qpos = f32(cpu.key_qpos)
        self.R_imu = f32(cpu.R_imu)
        self.gravity_dir = f32(cpu.gravity_dir)
        self.foot_ids = jnp.asarray(cpu.foot_ids)
        self.foot_body = jnp.asarray(m.geom_bodyid[cpu.foot_ids])
        self.foot_root = jnp.asarray(m.body_rootid[m.geom_bodyid[cpu.foot_ids]])
        self.geom_base = jnp.asarray(cpu.geom_base)
        self.geom_nonfoot = jnp.asarray(cpu.geom_nonfoot)
        self.fric_ids = jnp.asarray(np.concatenate([[cpu.floor_id], cpu.foot_ids]))
        self.base_id = cpu.base_id
        self.nom_friction = float(cpu.nom_friction[cpu.floor_id, 0])
        self.nom_kp = f32(cpu.nom_kp)
        self.nom_kv = f32(cpu.nom_kv)
        self.nom_forcerange = f32(cpu.nom_forcerange)
        self.nom_mass = float(cpu.nom_mass)
        self.nom_inertia = f32(cpu.nom_inertia)
        self.weights = jnp.asarray([cpu.weights[k] for k in REWARD_TERMS], dtype=jnp.float32)
        dr = self.cfg["domain_rand"]
        self.buf_len = (int(dr["action_delay_steps"][1]) + 1) if self.randomize else 1

        sc = self.cfg["obs"]["scales"]
        self.s_ang = sc["ang_vel"]
        self.s_cmd = f32(sc["cmd"])
        self.s_dvel = sc["dof_vel"]
        self.s_lin = sc["lin_vel"]

        self.reset = jax.jit(self._reset_batch)
        self.step = jax.jit(jax.vmap(self._step_one))

    def _uniform(self, key, lo_hi, shape=()):
        return jax.random.uniform(key, shape, minval=lo_hi[0], maxval=lo_hi[1])

    def _sample_params(self, key):
        if not self.randomize:
            return Params(
                friction=jnp.float32(self.nom_friction),
                mass_scale=jnp.float32(1.0),
                payload=jnp.float32(0.0),
                kp=self.nom_kp,
                kv=self.nom_kv,
                force_scale=jnp.ones(self.nu, dtype=jnp.float32),
                delay=jnp.int32(0),
            )
        dr = self.cfg["domain_rand"]
        k = jax.random.split(key, 7)
        lo, hi = dr["action_delay_steps"]
        return Params(
            friction=self._uniform(k[0], dr["friction"]),
            mass_scale=self._uniform(k[1], dr["base_mass_scale"]),
            payload=self._uniform(k[2], dr["payload_kg"]),
            kp=self.nom_kp * self._uniform(k[3], dr["kp_scale"], (self.nu,)),
            kv=self.nom_kv * self._uniform(k[4], dr["kv_scale"], (self.nu,)),
            force_scale=self._uniform(k[5], dr["torque_limit_scale"], (self.nu,)),
            delay=jax.random.randint(k[6], (), lo, hi + 1),
        )

    def _model(self, p):
        if not self.randomize:
            return self.mx
        mx = self.mx
        return mx.replace(
            geom_friction=mx.geom_friction.at[self.fric_ids, 0].set(p.friction),
            body_mass=mx.body_mass.at[self.base_id].set(self.nom_mass * p.mass_scale + p.payload),
            body_inertia=mx.body_inertia.at[self.base_id].set(self.nom_inertia * p.mass_scale),
            actuator_gainprm=mx.actuator_gainprm.at[:, 0].set(p.kp),
            actuator_biasprm=mx.actuator_biasprm.at[:, 1].set(-p.kp).at[:, 2].set(-p.kv),
            actuator_forcerange=self.nom_forcerange * p.force_scale[:, None],
        )

    def _sample_command(self, key, step_count):
        cc = self.cfg["commands"]
        k = jax.random.split(key, 5)
        c = jnp.stack([self._uniform(k[0], cc["vx"]), self._uniform(k[1], cc["vy"]), self._uniform(k[2], cc["yaw"])])
        c = jnp.where(jax.random.uniform(k[3]) < cc["zero_prob"], 0.0, c)
        small = jnp.linalg.norm(c[:2]) < cc["min_xy"]
        c = c.at[:2].set(jnp.where(small, 0.0, c[:2]))
        at = step_count + jnp.round(self._uniform(k[4], cc["resample_s"]) / self.dt).astype(jnp.int32)
        return c, at

    def _sample_push(self, key, step_count):
        if not self.randomize:
            return jnp.int32(np.iinfo(np.int32).max)
        iv = self.cfg["domain_rand"]["push_interval_s"]
        return step_count + jnp.round(self._uniform(key, iv) / self.dt).astype(jnp.int32)

    def _measure(self, d):
        q = d.qpos[self.ra + 3:self.ra + 7]
        lin = rotate_inv(q, d.qvel[self.rv:self.rv + 3])
        ang = d.qvel[self.rv + 3:self.rv + 6]
        grav = rotate_inv(q, self.gravity_dir)
        c = d._impl.contact
        g1, g2 = c.geom[:, 0], c.geom[:, 1]
        other = jnp.where(g1 == self.floor_id, g2, jnp.where(g2 == self.floor_id, g1, self.ngeom))
        active = (c.dist < 0.0).astype(jnp.int32)
        touching = jnp.zeros(self.ngeom + 1, dtype=jnp.int32).at[other].max(active)[: self.ngeom] > 0
        foot_contact = touching[self.foot_ids]
        cvel = d.cvel[self.foot_body]
        r = d.geom_xpos[self.foot_ids] - d.subtree_com[self.foot_root]
        v_foot = cvel[:, 3:] + jnp.cross(cvel[:, :3], r)
        return {
            "lin_vel": lin,
            "ang_vel": ang,
            "gravity": grav,
            "height": d.qpos[self.ra + 2] - self.floor_z,
            "dof_pos": d.qpos[self.qadr],
            "dof_vel": d.qvel[self.vadr],
            "foot_contact": foot_contact,
            "base_contact": jnp.any(touching & self.geom_base),
            "n_collisions": jnp.sum(touching & self.geom_nonfoot).astype(jnp.float32),
            "foot_slip": jnp.sum((v_foot[:, 0] ** 2 + v_foot[:, 1] ** 2) * foot_contact),
            "actuator_force": d.actuator_force,
        }

    def _obs(self, meas, command, last_action, friction, key):
        nc = self.cfg["obs"]["noise"]
        ang = self.R_imu.T @ meas["ang_vel"]
        grav = self.R_imu.T @ meas["gravity"]
        dpos = meas["dof_pos"] - self.default_q
        dvel = meas["dof_vel"]
        if self.noise_enabled:
            k = jax.random.split(key, 4)
            ang = ang + jax.random.uniform(k[0], (3,), minval=-nc["ang_vel"], maxval=nc["ang_vel"])
            grav = grav + jax.random.uniform(k[1], (3,), minval=-nc["gravity"], maxval=nc["gravity"])
            dpos = dpos + jax.random.uniform(k[2], (self.nu,), minval=-nc["dof_pos"], maxval=nc["dof_pos"])
            dvel = dvel + jax.random.uniform(k[3], (self.nu,), minval=-nc["dof_vel"], maxval=nc["dof_vel"])
        obs = jnp.concatenate([
            ang * self.s_ang,
            grav,
            command * self.s_cmd,
            dpos,
            dvel * self.s_dvel,
            last_action,
            meas["lin_vel"] * self.s_lin,
            meas["height"][None],
            meas["foot_contact"].astype(jnp.float32),
            friction[None],
        ])
        return jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(jnp.float32)

    def _reset_one(self, key):
        k = jax.random.split(key, 7)
        p = self._sample_params(k[0])
        n = self.cfg["episode"]["reset_joint_noise"]
        joints = jnp.clip(self.default_q + jax.random.uniform(k[1], (self.nu,), minval=-n, maxval=n), self.q_lo, self.q_hi)
        qpos = self.key_qpos.at[self.qadr].set(joints)
        if self.cfg["episode"]["random_yaw"]:
            yaw = jax.random.uniform(k[2], minval=-jnp.pi, maxval=jnp.pi)
            qy = jnp.stack([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)])
            qpos = qpos.at[self.ra + 3:self.ra + 7].set(quat_mul(qy, self.key_qpos[self.ra + 3:self.ra + 7]))
        d = self.d0.replace(
            qpos=qpos,
            qvel=jnp.zeros_like(self.d0.qvel),
            ctrl=jnp.clip(self.default_q, self.q_lo, self.q_hi),
            qacc_warmstart=jnp.zeros_like(self.d0.qacc_warmstart),
        )
        d = mjx.forward(self._model(p), d)
        zero = jnp.int32(0)
        command, resample_at = self._sample_command(k[3], zero)
        state = State(
            data=d,
            params=p,
            rng=k[6],
            command=command,
            resample_at=resample_at,
            push_at=self._sample_push(k[4], zero),
            step_count=zero,
            last_action=jnp.zeros(self.nu, dtype=jnp.float32),
            action_buf=jnp.zeros((self.buf_len, self.nu), dtype=jnp.float32),
            last_dof_vel=jnp.zeros(self.nu, dtype=jnp.float32),
            air_time=jnp.zeros(self.n_feet, dtype=jnp.float32),
            last_contacts=jnp.zeros(self.n_feet, dtype=bool),
        )
        meas = self._measure(d)
        obs = self._obs(meas, command, state.last_action, p.friction, k[5])
        return state, obs

    def _reset_batch(self, key):
        return jax.vmap(self._reset_one)(jax.random.split(key, self.num_envs))

    def _rewards(self, s, meas, a):
        rc = self.cfg["rewards"]
        sigma = rc["tracking_sigma"]
        sigma_yaw = rc.get("tracking_sigma_yaw", sigma)
        cmd = s.command
        lin_err2 = jnp.sum((cmd[:2] - meas["lin_vel"][:2]) ** 2)
        yaw_err = cmd[2] - meas["ang_vel"][2]

        contact = meas["foot_contact"]
        contact_filt = contact | s.last_contacts
        first_contact = (s.air_time > 0.0) & contact_filt
        air_time = s.air_time + self.dt
        air = jnp.sum((air_time - rc["feet_air_time_target"]) * first_contact)
        air = jnp.where(jnp.linalg.norm(cmd[:2]) > rc["feet_air_time_min_cmd"], air, 0.0)
        air_time = air_time * (~contact_filt)

        q = meas["dof_pos"]
        raw = {
            "tracking_lin_vel": jnp.exp(-lin_err2 / sigma),
            "tracking_ang_vel": jnp.exp(-yaw_err ** 2 / sigma_yaw),
            "feet_air_time": air,
            "lin_vel_z": meas["lin_vel"][2] ** 2,
            "ang_vel_xy": jnp.sum(meas["ang_vel"][:2] ** 2),
            "orientation": jnp.sum(meas["gravity"][:2] ** 2),
            "torques": jnp.sum(meas["actuator_force"] ** 2),
            "action_rate": jnp.sum((a - s.last_action) ** 2),
            "dof_acc": jnp.sum(((meas["dof_vel"] - s.last_dof_vel) / self.dt) ** 2),
            "collision": meas["n_collisions"],
            "foot_slip": meas["foot_slip"],
            "dof_pos_limits": jnp.sum(jnp.clip(self.soft_lo - q, 0.0, None) + jnp.clip(q - self.soft_hi, 0.0, None)),
        }
        terms = jnp.stack([raw[k] for k in REWARD_TERMS]) * self.weights * self.dt
        return terms, jnp.sqrt(lin_err2), jnp.abs(yaw_err), air_time, contact

    def _step_one(self, s, action):
        p = s.params
        mi = self._model(p)
        k = jax.random.split(s.rng, 6)
        clip = self.cfg["control"]["action_clip"]
        a = jnp.clip(action.astype(jnp.float32), -clip, clip)
        buf = jnp.concatenate([a[None], s.action_buf[:-1]], axis=0)
        applied = buf[p.delay]
        ctrl = jnp.clip(self.default_q + self.cfg["control"]["action_scale"] * applied, self.q_lo, self.q_hi)
        d = s.data.replace(ctrl=ctrl)

        push_at = s.push_at
        if self.randomize:
            do_push = s.step_count >= s.push_at
            v = self.cfg["domain_rand"]["push_vel_xy"]
            kick = jax.random.uniform(k[0], (2,), minval=-v, maxval=v) * do_push
            d = d.replace(qvel=d.qvel.at[self.rv:self.rv + 2].add(kick))
            push_at = jnp.where(do_push, self._sample_push(k[5], s.step_count), s.push_at)

        d = jax.lax.fori_loop(0, self.decimation, lambda _, dd: mjx.step(mi, dd), d)
        step_count = s.step_count + 1
        meas = self._measure(d)

        terms, lin_err, yaw_err, air_time, contacts = self._rewards(s, meas, a)
        terms = jnp.where(jnp.all(jnp.isfinite(terms)), terms, 0.0)
        reward = jnp.sum(terms)
        if self.cfg["rewards"]["only_positive_rewards"]:
            reward = jnp.maximum(reward, 0.0)

        ep = self.cfg["episode"]
        cos_tilt = -meas["gravity"][2]
        fell = (
            meas["base_contact"]
            | (cos_tilt < np.cos(np.deg2rad(ep["fall_tilt_deg"])))
            | (meas["height"] < ep["min_base_height"])
            | ~jnp.all(jnp.isfinite(d.qpos))
        )
        term_rew = jnp.where(fell, self.cfg["rewards"]["termination_penalty"], 0.0)
        reward = reward + term_rew
        truncated = (~fell) & (step_count >= self.max_steps)

        new_cmd, new_at = self._sample_command(k[1], step_count)
        resample = step_count >= s.resample_at
        command = jnp.where(resample, new_cmd, s.command)
        resample_at = jnp.where(resample, new_at, s.resample_at)

        obs = self._obs(meas, command, a, p.friction, k[2])
        nxt = State(
            data=d,
            params=p,
            rng=k[3],
            command=command,
            resample_at=resample_at,
            push_at=push_at,
            step_count=step_count,
            last_action=a,
            action_buf=buf,
            last_dof_vel=meas["dof_vel"],
            air_time=air_time,
            last_contacts=contacts,
        )
        info = jnp.concatenate([terms, jnp.stack([term_rew, lin_err, yaw_err])]).astype(jnp.float32)

        done = fell | truncated
        reset_state, reset_obs = self._reset_one(k[4])
        out_state = jax.tree_util.tree_map(lambda r, n: jnp.where(done, r, n), reset_state, nxt)
        out_obs = jnp.where(done, reset_obs, obs)
        return out_state, out_obs, obs, reward.astype(jnp.float32), fell.astype(jnp.float32), truncated.astype(jnp.float32), info
