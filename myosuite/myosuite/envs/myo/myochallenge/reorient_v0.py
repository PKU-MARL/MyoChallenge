""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
from traceback import print_tb
import numpy as np
import gym

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import mat2euler, euler2quat, mat2quat, mulQuat, negQuat

class ReorientEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot', 'rot_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist": 10.0,
        "rot_dist": 1.0,
        "solved": 250.0,
        "act_reg": 0.01,
        "const": 0.0
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        self._setup(**kwargs)

    def _setup(self,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            goal_pos = (0.0, 0.0),      # goal position range (relative to initial pos)
            goal_rot = (.785, .785),    # goal rotation range (relative to initial rot)
            obj_size_change = 0,        # object size change (relative to initial size)
            obj_friction_change = (0,0,0),# object friction change (relative to initial size)
            pos_th = .025,              # position error threshold
            rot_th = 0.262,             # rotation error threshold
            drop_th = .200,             # drop height threshold
            **kwargs,
        ):
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.goal_sid = self.sim.model.site_name2id("target_o")
        self.success_indicator_sid = self.sim.model.site_name2id("target_ball")
        self.goal_bid = self.sim.model.body_name2id("target")
        self.goal_init_pos = self.sim.data.site_xpos[self.goal_sid].copy()
        self.goal_obj_offset = self.sim.data.site_xpos[self.goal_sid]-self.sim.data.site_xpos[self.object_sid] # visualization offset between target and object
        self.goal_pos = goal_pos
        self.goal_rot = goal_rot
        self.pos_th = pos_th
        self.rot_th = rot_th
        self.drop_th = drop_th
        self.max_episode_steps = 150
        self.counter = 0
        self.accum_solve = 0

        # custom sites
        self.my_tip_sids = []
        tip_site_name=['THtip', 'IFtip', 'MFtip', 'RFtip', 'LFtip']
        for site in tip_site_name:
                self.my_tip_sids.append(self.sim.model.site_name2id(site))

        # setup for object randomization
        self.target_gid = self.sim.model.geom_name2id('target_dice')
        self.target_default_size = self.sim.model.geom_size[self.target_gid].copy()

        object_bid = self.sim.model.body_name2id('Object')
        self.object_gid0 = self.sim.model.body_geomadr[object_bid]
        self.object_gidn = self.object_gid0 + self.sim.model.body_geomnum[object_bid]
        self.object_default_size = self.sim.model.geom_size[self.object_gid0:self.object_gidn].copy()
        self.object_default_pos = self.sim.model.geom_pos[self.object_gid0:self.object_gidn].copy()

        self.obj_size_change = {'high':np.ones((1,))*obj_size_change, 'low':np.ones((1,))*(-obj_size_change)}
        self.obj_friction_range = {'high':self.sim.model.geom_friction[self.object_gid0:self.object_gidn] + obj_friction_change,
                                    'low':self.sim.model.geom_friction[self.object_gid0:self.object_gidn] - obj_friction_change}

        self.friction = self.np_random.uniform(**self.obj_friction_range)
        self.del_size = self.np_random.uniform(**self.obj_size_change)

        # asymmetric observation
        DEFAULT_ASM_KEYS = ['friction', 'del_size', 'obj_vel', 'tip_pos']
        self.asm_keys = DEFAULT_ASM_KEYS

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        self.init_qpos[:-7] *= 0 # Use fully open as init pos
        self.init_qpos[0] = -1.5 # Palm up

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        t, asm_obs = self.obsdict2obsvec(self.obs_dict, self.asm_keys)
        shared = obs.shape[-1]
        dedicate = asm_obs.shape[-1]
        self.observation_space = gym.spaces.Box(-1*np.ones(shared+dedicate), 1*np.ones(shared+dedicate), dtype=np.float32)
        self.asym_observation_space = gym.spaces.Box(-1*np.ones(dedicate), 1*np.ones(dedicate), dtype=np.float32)

    def update_dr(self, **kwargs) :

        for key, val in kwargs.items() :
            if key == "max_episode_steps" :
                self.max_episode_steps = val
                # print("max_episode_steps updated to {}".format(val))
            elif key == "goal_pos" :
                self.goal_pos = val
            elif key == "goal_rot" :
                self.goal_rot = val
            elif key == "obj_size_change" :
                self.obj_size_change = {
                    'high':np.ones((1,))*val,
                    'low':np.ones((1,))*(-val)
                }
            elif key == "obj_friction_change" :
                self.obj_friction_change = {
                    'high': np.ones((15, 3)) * np.array([1.0, 0.005, 0.00005]) + np.array(val),
                    'low':np.ones((15, 3)) * np.array([1.0, 0.005, 0.00005]) - np.array(val)
                }
            else :
                raise ValueError("Invalid key for update_dr")

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:-7].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:-6].copy()*self.dt
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['goal_pos'] = sim.data.site_xpos[self.goal_sid]
        obs_dict['pos_err'] = obs_dict['goal_pos'] - obs_dict['obj_pos'] - self.goal_obj_offset # correct for visualization offset between target and object
        obs_dict['obj_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.object_sid],(3,3)))
        obs_dict['goal_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.goal_sid],(3,3)))
        obs_dict['obj_quat'] = mat2quat(np.reshape(sim.data.site_xmat[self.object_sid],(3,3)))
        obs_dict['goal_quat'] = mat2quat(np.reshape(sim.data.site_xmat[self.goal_sid],(3,3)))
        obs_dict['rot_err'] = obs_dict['goal_rot'] - obs_dict['obj_rot']

        # god-perspective observation
        obs_dict['friction'] = self.friction
        obs_dict['del_size'] = self.del_size
        obs_dict['obj_vel'] = self.sim.data.site_xvelp[self.object_sid]
        tip_pos = np.array([])
        for isite in range(len(self.my_tip_sids)):
            tip_pos = np.append(tip_pos, self.sim.data.site_xpos[self.my_tip_sids[isite]].copy())
        tip_pos = tip_pos.reshape(-1, 3)
        obs_dict['tip_pos'] = tip_pos

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):

        quat_diff = mulQuat(self.obs_dict['goal_quat'][0][0], negQuat(self.obs_dict['obj_quat'][0][0]))
        rot_dist = 2.0 * np.arcsin(np.clip(np.linalg.norm(quat_diff[:3], axis=-1), a_min=0.0, a_max=1.0))

        pos_dist = float(np.abs(np.linalg.norm(self.obs_dict['pos_err'], axis=-1)))
        # rot_dist = float(np.abs(np.linalg.norm(self.obs_dict['rot_err'], axis=-1)))
        act_mag = float(np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0)

        obj_vel = self.obs_dict['obj_vel']
        tip_pos = self.obs_dict['tip_pos']

        # print(np.linalg.norm(tip_pos - self.obs_dict['obj_pos'].reshape(1, 3), axis=-1))

        obj_vel = np.abs(np.linalg.norm(obj_vel, axis=-1))
        tip_err = float(np.sum(np.linalg.norm(tip_pos - self.obs_dict['obj_pos'].reshape(1, 3), axis=-1)))
        drop = float(pos_dist > self.drop_th)


        # a_pos = 3.0 / 1600
        # a_rot = 0.20593200000000003
        a_pos = 3./100
        a_rot = 0.2

        solved = (pos_dist<self.pos_th) and (rot_dist<self.rot_th) and (not drop)

        rwd_dict = collections.OrderedDict((
            # Perform reward tuning here --
            # Update Optional Keys section below
            # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
            # Examples: Env comes pre-packaged with two keys pos_dist and rot_dist

            # Optional Keys
            ('pos_dist', -pos_dist),
            ('rot_dist', 1.0/(np.abs(rot_dist) + 0.1)),
            ('obj_vel', -1.*obj_vel),
            ('drop', -1.*drop),
            ('tip_err', -1.*tip_err),
            ('const', 1.0),
            # Must keys
            ('act_reg', -1.*act_mag),
            ('sparse', -rot_dist-10.0*pos_dist),
            ('solved', solved),
            ('done', drop or (self.counter >= self.max_episode_steps)),
        ))

        self.accum_solve += solved
        rew_list = [wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()]
        rwd_dict['dense'] = np.sum(np.array(rew_list, dtype=object), axis=0)

        # Sucess Indicator
        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([0, 2]) if rwd_dict['solved'] else np.array([2, 0])
        return rwd_dict


    def get_metrics(self, paths, successful_steps=5):
        """
        Evaluate paths and report metrics
        """
        num_success = 0
        num_paths = len(paths)

        # average sucess over entire env horizon
        for path in paths:
            # record success if solved for provided successful_steps
            if np.sum(path['env_infos']['rwd_dict']['solved'] * 1.0) > successful_steps:
                num_success += 1
        score = num_success/num_paths

        # average activations over entire trajectory (can be shorter than horizon, if done) realized
        effort = -1.0*np.mean([np.mean(p['env_infos']['rwd_dict']['act_reg']) for p in paths])

        metrics = {
            'score': score,
            'effort':effort,
            }
        return metrics

    def reset_target(self) :

        self.sim.model.body_pos[self.goal_bid] = self.goal_init_pos + \
            self.np_random.uniform( high=self.goal_pos[1], low=self.goal_pos[0], size=3)

        self.sim.model.body_quat[self.goal_bid] = \
            euler2quat(self.np_random.uniform(high=self.goal_rot[1], low=self.goal_rot[0], size=3))

        # Die friction changes
        self.friction = self.np_random.uniform(**self.obj_friction_range)
        self.sim.model.geom_friction[self.object_gid0:self.object_gidn] = self.friction

        # Die and Target size changes
        self.del_size = self.np_random.uniform(**self.obj_size_change)
        # adjust size of target
        self.sim.model.geom_size[self.target_gid] = self.target_default_size + self.del_size
        # adjust size of die
        self.sim.model.geom_size[self.object_gid0:self.object_gidn-3][:,1] = self.object_default_size[:-3][:,1] + self.del_size
        self.sim.model.geom_size[self.object_gidn-3:self.object_gidn] = self.object_default_size[-3:] + self.del_size
        # adjust boundary of die
        object_gpos = self.sim.model.geom_pos[self.object_gid0:self.object_gidn]
        self.sim.model.geom_pos[self.object_gid0:self.object_gidn] = object_gpos/abs(object_gpos+1e-16) * (abs(self.object_default_pos) + self.del_size)

    def reset(self):
        # reset the counter for the number of steps
        self.counter = 0

        self.reset_target()

        obs = super().reset()
        t, asm_obs = self.obsdict2obsvec(self.obs_dict, self.asm_keys)
        return np.concatenate((obs, asm_obs))

    def step(self, a):

        self.counter += 1

        if self.accum_solve >= 5 :
            self.accum_solve = 0
            self.reset_target()
        obs, rewards, dones, info = super().step(a)
        t, asm_obs = self.obsdict2obsvec(self.obs_dict, self.asm_keys)
        obs = np.concatenate((obs, asm_obs))
        return obs, rewards, dones, info
