import numpy as np
from transformation import r2rpy
import mujoco as mj

class SnapbotGymClass:
    """MuJoCo Snapbot gym – high-jump version (완전 개선판)"""

    def __init__(self, env, HZ=50,
                history_total_sec=2.0, history_intv_sec=0.1,
                VERBOSE=True):
        self.env = env
        self.HZ, self.dt = HZ, 1 / HZ
        self.name = env.name

        # 1단계: 기본 변수들 먼저 초기화
        self.phase = 'crouch'
        self.tick = 0
        self.phase_tick = 0
        self.mujoco_nstep = max(1, int(self.env.HZ / self.HZ))
        self.contact_thr = 0.5
        
        # 2단계: get_state()에서 필요한 변수들 초기화
        self.crouch_phase_ticks = int(1.5 * HZ)
        self.push_phase_ticks = int(1.5 * HZ)
        self.air_phase_ticks = int(1.5 * HZ)
        self.state_update_count = 0
        
        # 3단계: 임시 더미 상태로 차원 계산
        # 임시 초기화 (get_state() 호출용)
        self.state_mean = np.zeros(1)  # 임시 더미 값
        self.state_std = np.ones(1)    # 임시 더미 값
        
        # 실제 상태 차원 계산
        sample_state = self.get_state()
        self.state_dim = len(sample_state)
        
        # 4단계: 올바른 차원으로 재초기화
        self.state_mean = np.zeros(self.state_dim)
        self.state_std = np.ones(self.state_dim)
        self.state_update_count = 0  # 재설정

        # History buffer 설정
        self.n_history = int(HZ * history_total_sec)
        self.history_intv_tick = int(HZ * history_intv_sec)
        self.history_ticks = np.arange(0, self.n_history, self.history_intv_tick)
        
        self.state_history = np.zeros((self.n_history, self.state_dim))
        self.tick_history = np.zeros((self.n_history, 1))
        self.o_dim = self.state_history[self.history_ticks, :].size
        
        self.knee_slots = [1, 3, 5, 7]
        self.knee_dirs = np.array([1, 1, 1, 1], dtype=float)
        self.a_dim  = len(self.knee_slots)
        self.action_prev = np.zeros(self.a_dim)

        # 점프 변수 초기화
        self.h_base = self.env.get_p_body('torso')[2]
        self.h_max = self.h_base
        self.action_prev = np.zeros(self.a_dim)
        self.h_max_ep = 0.0
        self.air_time_ep = 0.0
        self.done_flag = False
        self.foot_sensor_idxs = [
            i for i, n in enumerate(self.env.sensor_names)
            if n.endswith('_4')
        ]

        if VERBOSE:
            print(f'[Snapbot Enhanced] a_dim:{self.a_dim} o_dim:{self.o_dim}')
            print(f'Crouch phase: {self.crouch_phase_ticks} ticks')
            print(f'Push phase: {self.push_phase_ticks} ticks')


    def get_state(self):
        """개선된 상태 관찰 함수"""
        qpos = self.env.data.qpos[self.env.ctrl_qpos_idxs]
        qvel = self.env.data.qvel[self.env.ctrl_qvel_idxs]
        
        # 관절 속도 클리핑 및 스케일링 개선[1]
        qvel_scaled = np.clip(qvel / 5.0, -2.0, 2.0)
        
        R_t = self.env.get_R_body('torso').reshape(-1)
        p_t = self.env.get_p_body('torso')
        torso_h = np.array([p_t[2]])
        torso_y = np.array([p_t[1]])

        # 센서 값 처리 개선[1]
        sensor_values = self.env.get_sensor_values(sensor_names=self.env.sensor_names)
        contact = (sensor_values > self.contact_thr).astype(float)

        # 현재 단계 정보 추가[2]
        phase_info = np.array([
            float(self.tick < self.crouch_phase_ticks),  # crouch phase
            float(self.crouch_phase_ticks <= self.tick < 
                (self.crouch_phase_ticks + self.push_phase_ticks)),  # push phase
            float(self.tick >= (self.crouch_phase_ticks + self.push_phase_ticks))  # air phase
        ])

        # 상태 구성
        raw_state = np.concatenate([
            qpos, qvel_scaled, R_t, torso_h, torso_y, contact, phase_info
        ])

        # 온라인 정규화 적용
        if self.state_update_count > 100:  # 충분한 샘플 후 정규화 시작
            normalized_state = (raw_state - self.state_mean) / (self.state_std + 1e-8)
        else:
            normalized_state = raw_state
            # 상태 통계 업데이트
            self.state_mean = ((self.state_mean * self.state_update_count + raw_state) / 
                            (self.state_update_count + 1))
            self.state_std = np.sqrt(((self.state_std ** 2 * self.state_update_count + 
                                     (raw_state - self.state_mean) ** 2) / 
                                    (self.state_update_count + 1)))
            self.state_update_count += 1

        return np.nan_to_num(normalized_state, nan=0.0, posinf=1.0, neginf=-1.0)

    def get_observation(self):
        return self.state_history[self.history_ticks, :].flatten()

    def sample_action(self):
        a_min = self.env.ctrl_ranges[self.knee_slots, 0]
        a_max = self.env.ctrl_ranges[self.knee_slots, 1]
        return a_min + (a_max - a_min) * np.random.rand(self.a_dim)

    def _feet_z(self):
        """발 끝 높이 계산"""
        return np.array([self.env.get_p_body(b)[2] for b in
                         ['Leg_module_1_4', 'Leg_module_2_4',
                          'Leg_module_4_4', 'Leg_module_5_4']])

    def _foot_contact(self):
        """접촉 감지 개선"""
        vals = self.env.get_sensor_values(sensor_names=self.env.sensor_names)
        return np.any(vals[self.foot_sensor_idxs] > self.contact_thr)
    
    def _feet_contact_ratio(self):
        vals = self.env.get_sensor_values(sensor_names=self.env.sensor_names[:4])  # 발끝 4개
        return np.mean(vals > self.contact_thr)

    def _foot_normal_z(self):
        """발바닥 법선 벡터 계산"""
        bodies = ['Leg_module_1_4', 'Leg_module_2_4',
                  'Leg_module_4_4', 'Leg_module_5_4']
        return np.array([np.dot(self.env.get_R_body(b)[:, 2], [0, 0, 1])
                         for b in bodies])

    def _leg_axis_z(self):
        """다리 축 수직성 계산"""
        bodies = ['Leg_module_1_2', 'Leg_module_2_2',
                'Leg_module_4_2', 'Leg_module_5_2']
        return np.array([abs(np.dot(self.env.get_R_body(b)[:, 2], [0, 0, 1]))
                        for b in bodies])
    
    def _pad_action(self, a_knee):
        a_full = np.zeros(len(self.env.ctrl_qpos_idxs))
        a_full[self.knee_slots] = a_knee
        return a_full

    def _reward_jump(self, h_cur, v_z, contact, roll_pen):
        """점프 보상 함수 개선[5]"""
        k_h, k_peak, k_v, k_land = 3.0, 15.0, 1.5, 8.0
        
        # 높이 보상 (제곱으로 강화)
        height_gain = max(0, h_cur - self.h_base)
        r = k_h * (height_gain ** 1.5)
        
        # 상향 속도 보상
        r += k_v * max(0, v_z) * 0.5
        
        # 에피소드 최고점 보상
        if self.done_flag:
            peak_reward = k_peak * max(0, self.h_max_ep - self.h_base)
            r += peak_reward
            
        # 착지 자세 보상
        if contact:
            upright = np.dot(self.env.get_R_body('torso')[:, 2], [0, 0, 1])
            r += k_land * (upright ** 2)
            
        return np.clip(r, -15, 15)
    
    def _joint_angle_error(self, q_cur, q_goal, idxs):
        angles = q_cur[idxs]
        error = np.abs(angles - q_goal)
        return error

    def _reward_energy(self, torque, action):
        """에너지 효율성 보상"""
        return (-3e-4 * np.sum(torque ** 2) - 
                8e-3 * np.sum((action - self.action_prev) ** 2))

    def step(self, action, max_time=np.inf):
        """무릎 관절(4 DoF)만으로 점프하도록 수정한 step 함수"""

        # --------------------------------------------------
        # 0) 준비
        # --------------------------------------------------
        self.tick += 1
        self.phase_tick += 1
        r_sync = 0.0          # 디버깅용 기록 변수
        r_knee = 0.0
        r_takeoff = 0.0

        # --------------------------------------------------
        # 1) 행동(4-D) → 안전 클리핑 → 8-D 패딩
        # --------------------------------------------------
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)     # (4,)
        lo = self.env.ctrl_ranges[self.knee_slots, 0]
        hi = self.env.ctrl_ranges[self.knee_slots, 1]
        action_knee = np.clip(action, lo, hi)                                # (4,)

        action_full = np.zeros(len(self.env.ctrl_qpos_idxs))                 # (8,)
        action_full[self.knee_slots] = action_knee

        # --------------------------------------------------
        # 2) 이전 상태 저장
        # --------------------------------------------------
        p_prev = self.env.get_p_body('torso')
        q_prev = self.env.data.qpos[self.env.ctrl_qpos_idxs].copy()
        contact_prev = self._foot_contact()

        # --------------------------------------------------
        # 3) 시뮬레이션 실행
        # --------------------------------------------------
        self.env.step(ctrl=action_full, nstep=self.mujoco_nstep)

        # --------------------------------------------------
        # 4) 현재 상태 계산
        # --------------------------------------------------
        p_cur  = self.env.get_p_body('torso')
        R_cur  = self.env.get_R_body('torso')
        q_cur  = self.env.data.qpos[self.env.ctrl_qpos_idxs].copy()
        contact_cur = self._foot_contact()

        # --------------------------------------------------
        # 5) 종료 조건
        # --------------------------------------------------
        ROLLOVER      = np.dot(R_cur[:, 2], [0, 0, 1]) < 0.15
        OUT_OF_BOUNDS = abs(p_cur[0]) > 5.0 or abs(p_cur[1]) > 5.0
        self.done_flag = done = (self.get_sim_time() >= max_time) or ROLLOVER or OUT_OF_BOUNDS

        # --------------------------------------------------
        # 6) 동역학 & 기록
        # --------------------------------------------------
        h_cur  = p_cur[2]
        v_z    = (h_cur - p_prev[2]) / self.dt
        self.h_max     = max(self.h_max, h_cur)
        self.h_max_ep  = max(self.h_max_ep, h_cur)
        if not contact_cur:
            self.air_time_ep += self.dt

        roll_pen   = 1 - np.dot(R_cur[:, 2], [0, 0, 1])
        joint_vel  = (q_cur - q_prev) / self.dt
        knee_vel   = joint_vel[self.knee_slots]                       # (4,)
        torque     = self.env.data.actuator_force.flatten()[self.knee_slots]
        target_z    = self.h_base - 0.03
        knee_ang = -q_cur[self.knee_slots] * self.knee_dirs

        # -------------------------------------------------
        # 6.5) phase transfer
        # -------------------------------------------------
        if self.phase == 'crouch':
            crouched = (h_cur <= target_z) or np.mean(knee_ang) > 0.5
            if crouched and self.phase_tick >= self.crouch_phase_ticks:
                self.phase = 'push'
                self.phase_tick = 0
        elif self.phase == 'push':
            if (self._feet_contact_ratio() < 0.1) and (v_z > 0):
                self.phase = 'air'
                self.phase_tick = 0
            elif self.phase_tick >= self.push_phase_ticks:
                self.phase = 'crouch'
                self.phase_tick = 0
        elif self.phase == 'air':
            if contact_cur and v_z <= 0 or self.phase_tick >= self.air_phase_ticks:
                self.phase = 'crouch'
                self.phase_tick = 0

        # --------------------------------------------------
        # 7) 단계별 보상
        # --------------------------------------------------
        r_phase = 0.0
        if self.phase == 'crouch':                       # Phase 0: crouch
            depth_err = abs(h_cur - target_z)

            # knee_ang = -q_cur[self.knee_slots] * self.knee_dirs
            sync_err = np.var(knee_ang)

            r_phase = (
                -5.0 * depth_err
                -2.0 * sync_err
                -0.01 * self.phase_tick
            )

        elif self.phase == 'push':  # Phase 1: push
            # -------------------------------------------------
            # (a) 무릎 펴는 속도 보상
            ext_vel     = self.knee_dirs * knee_vel          # + : 펴는 방향
            speed_mean  = np.mean(np.clip(ext_vel, 0, None))
            speed_var   = np.var(ext_vel)
            r_speed     = 80.0 * speed_mean - 30.0 * speed_var

            # -------------------------------------------------
            # (b) 무릎 각도 진행률 보상
            ctrl_hi = self.env.ctrl_ranges[self.knee_slots, 1]
            knee_goal = ctrl_hi - 0.03     
            err_knee    = self.knee_dirs * (knee_goal - q_cur[self.knee_slots])
            r_knee      = 6.0 * np.mean(np.clip(err_knee, 0, None))

            # -------------------------------------------------
            # (c) **너무 낮게 앉아 있으면 패널티*

            # -------------------------------------------------
            # (d) 이륙 순간 보너스
            if (contact_prev and not contact_cur and v_z > 0): 
                r_takeoff = 100.0 * v_z * (1.0 + max(0.0, h_cur - self.h_base))

                h_gain = h_cur - self.h_base
                r_takeoff += 300*h_gain
            else: 
                h_gain = h_cur - self.h_base
                r_takeoff += 100*h_gain

            # -------------------------------------------------
            # 총 push-phase 보상
            r_phase  = r_speed + r_knee + r_takeoff   # (+ r_n)
            r_sync   = r_speed     # 디버깅용

        elif self.phase == 'air':                                                   # Phase 2: air / landing
            r_jump    = 5.0 * self._reward_jump(h_cur, v_z, contact_cur, roll_pen)
            r_landing = 10.0 * max(0.0, np.dot(R_cur[:, 2], [0, 0, 1]))
            r_phase   = r_jump + r_landing

        # --------------------------------------------------
        # 8) 기타 보상
        # --------------------------------------------------
        # r_energy    = self._reward_energy(torque, action_knee)      # 4-D 사용
        r_collision = -15.0 if len(self.env.get_contact_info(must_exclude_prefix='floor')[2]) else 0.0
        r_survive   = -15.0 if ROLLOVER else 0.02

        reward = float(np.clip(r_phase + r_collision + r_survive, -300, 300))

        # --------------------------------------------------
        # 9) 기록 및 반환
        # --------------------------------------------------
        self.accumulate_state_history()
        phase_name = ('crouch' if self.tick < self.crouch_phase_ticks else
                    'push'  if self.tick < (self.crouch_phase_ticks + self.push_phase_ticks)
                    else 'air')

        info = dict(h_curr      = h_cur,
                    h_max_ep    = self.h_max_ep,
                    air_time_ep = self.air_time_ep,
                    phase       = phase_name,
                    contact     = contact_cur,
                    v_z         = v_z,
                    r_phase     = r_phase,
                    r_sync      = r_sync,
                    r_knee      = r_knee,
                    r_takeoff   = r_takeoff)

        self.action_prev = action_knee.copy()
        return self.get_observation(), reward, done, info

    def reset(self):
        """리셋 함수"""
        self.tick = 0
        self.env.reset(step=True)
        self.state_history[:] = 0.0
        self.tick_history[:] = 0.0

        self.h_base = self.env.get_p_body('torso')[2]
        self.h_max = self.h_base
        self.h_max_ep = 0.0
        self.air_time_ep = 0.0
        self.action_prev = np.zeros(self.a_dim)
        self.done_flag = False
        
        return self.get_observation()

    def render(self, TRACK_TORSO=True, PLOT_WORLD_COORD=True, 
            PLOT_TORSO_COORD=True, PLOT_SENSOR=True, 
            PLOT_CONTACT=True, PLOT_TIME=True, PLOT_DEBUG=True, PLOT_JOINT_LABEL=True):
        """개선된 렌더링 함수"""
        
        if TRACK_TORSO:
            p_lookat = self.env.get_p_body('torso')
            self.env.set_viewer(lookat=p_lookat)
            
        if PLOT_WORLD_COORD:
            self.env.plot_T(p=np.zeros(3), R=np.eye(3, 3), 
                        plot_axis=True, axis_len=1.0, axis_width=0.0025)
            
        if PLOT_TORSO_COORD:
            p_torso, R_torso = self.env.get_pR_body(body_name='torso')
            self.env.plot_T(p=p_torso, R=R_torso, plot_axis=True, 
                        axis_len=0.25, axis_width=0.0025)
            
        if PLOT_SENSOR:
            sensor_vals = self.env.get_sensor_values(sensor_names=self.env.sensor_names)
            contact_idxs = np.where(sensor_vals > self.contact_thr)[0]
            for idx in contact_idxs:
                sensor_name = self.env.sensor_names[idx]
                p_sensor = self.env.get_p_sensor(sensor_name)
                intensity = min(1.0, sensor_vals[idx] / 2.0)
                self.env.plot_sphere(p=p_sensor, r=0.02, 
                                   rgba=[1, 0, 0, 0.3 + 0.7 * intensity])
                
        if PLOT_CONTACT:
            self.env.plot_contact_info()
            
        if PLOT_TIME:
            p_torso, R_torso = self.env.get_pR_body(body_name='torso')
            self.env.plot_T(p=p_torso + 0.25 * R_torso[:, 2], R=np.eye(3, 3),
                        plot_axis=False, 
                        label='[%.2f]sec Tick:%d' % (self.env.get_sim_time(), self.tick))
            
        # 디버그 정보 표시
        if PLOT_DEBUG:
            phase_name = ('CROUCH' if self.tick < self.crouch_phase_ticks else
                        'PUSH' if self.tick < (self.crouch_phase_ticks + self.push_phase_ticks)
                        else 'AIR')
            debug_text = f"Phase: {phase_name} | H_max: {self.h_max_ep:.2f}m"
            p_torso, R_torso = self.env.get_pR_body(body_name='torso')
            self.env.plot_T(p=p_torso + 0.35 * R_torso[:, 2], R=np.eye(3, 3),
                        plot_axis=False, label=debug_text)
            
        if PLOT_JOINT_LABEL:
            for slot, qadr in enumerate(self.env.ctrl_qpos_idxs):
                # 1) qadr → joint index
                jidx   = next(j for j, adr in enumerate(self.env.model.jnt_qposadr)
                            if adr == qadr)
                j_name = self.env.joint_names[jidx]

                # 2) joint 가 붙어 있는 body 이름 얻기
                body_id   = int(self.env.model.jnt_bodyid[jidx])
                body_name = mj.mj_id2name(self.env.model,
                                        mj.mjtObj.mjOBJ_BODY,
                                        body_id)

                # mj_id2name() 반환값 정규화 (str or bytes or None)
                if body_name is None:
                    body_name = j_name
                elif isinstance(body_name, bytes):        # 구버전 대응
                    body_name = body_name.decode()

                # 3) 관절(앵커) 위치
                p_joint = (self.env.get_p_joint(j_name)
                        if hasattr(self.env, "get_p_joint")
                        else self.env.get_p_body(body_name))

                # 4) 라벨 위치 살짝 올리기
                p_label = p_joint + np.array([0, 0, 0.03])

                # 5) 텍스트 렌더
                self.env.plot_T(p=p_label,
                                R=np.eye(3),
                                plot_axis=False,
                                label=f"{slot}:{j_name}")
        
        self.env.render()

    # 나머지 유틸리티 함수들
    def init_viewer(self):
        self.env.init_viewer(distance=3.5)
        
    def close_viewer(self):
        self.env.close_viewer()
        
    def get_sim_time(self):
        return self.env.get_sim_time()
    
    def is_viewer_alive(self):
        return self.env.is_viewer_alive()
    
    def accumulate_state_history(self):
        state = self.get_state()
        self.state_history[1:, :] = self.state_history[:-1, :]
        self.state_history[0, :] = state
        self.tick_history[1:, :] = self.tick_history[:-1, :]
        self.tick_history[0, :] = self.tick
        
    def viewer_pause(self):
        self.env.viewer_pause()
        
    def grab_image(self, resize_rate=1.0, interpolation=0):
        return self.env.grab_image(rsz_rate=resize_rate, interpolation=interpolation)
