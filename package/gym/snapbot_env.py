import numpy as np
""" 
    Assume that the main notebook called 'sys.path.append('../../package/helper/')'
"""
from transformation import r2rpy

class SnapbotGymClass():
    """ 
        Snapbot gym 
    """
    def __init__(self,env,HZ=50,history_total_sec=2.0,history_intv_sec=0.1,VERBOSE=True):
        """
            Initialize
        """
        self.env               = env # MuJoCo environment
        self.HZ                = HZ
        self.dt                = 1/self.HZ
        self.history_total_sec = history_total_sec # history in seconds
        self.n_history         = int(self.HZ*self.history_total_sec) # number of history
        self.history_intv_sec  = history_intv_sec
        self.history_intv_tick = int(self.HZ*self.history_intv_sec) # interval between state in history
        self.history_ticks     = np.arange(0,self.n_history,self.history_intv_tick)
        
        self.mujoco_nstep      = self.env.HZ // self.HZ # nstep for MuJoCo step
        self.VERBOSE           = VERBOSE
        self.tick              = 0
        
        self.name              = env.name
        self.state_prev        = self.get_state()
        self.action_prev       = self.sample_action()
        
        # Dimensions
        self.state_dim         = len(self.get_state())
        self.state_history     = np.zeros((self.n_history,self.state_dim))
        self.tick_history      = np.zeros((self.n_history,1))
        self.o_dim             = len(self.get_observation())
        self.a_dim             = env.n_ctrl
        
        if VERBOSE:
            print ("[%s] Instantiated"%
                   (self.name))
            print ("   [info] dt:[%.4f] HZ:[%d], env-HZ:[%d], mujoco_nstep:[%d], state_dim:[%d], o_dim:[%d], a_dim:[%d]"%
                   (self.dt,self.HZ,self.env.HZ,self.mujoco_nstep,self.state_dim,self.o_dim,self.a_dim))
            print ("   [history] total_sec:[%.2f]sec, n:[%d], intv_sec:[%.2f]sec, intv_tick:[%d]"%
                   (self.history_total_sec,self.n_history,self.history_intv_sec,self.history_intv_tick))
            print ("   [history] ticks:%s"%(self.history_ticks))
        
    def get_state(self):
        """
            Get state (33)
            : Current state consists of 
                1) current joint position (8)
                2) current joint velocity (8)
                3) torso rotation (9)
                4) torso height (1)
                5) torso y value (1)
                6) contact info (8)
        """
        # Joint position
        qpos = self.env.data.qpos[self.env.ctrl_qpos_idxs] # joint position
        # Joint velocity
        qvel = self.env.data.qvel[self.env.ctrl_qvel_idxs] # joint velocity
        # Torso rotation matrix flattened
        R_torso_flat = self.env.get_R_body(body_name='torso').reshape((-1)) # torso rotation
        # Torso height
        p_torso = self.env.get_p_body(body_name='torso') # torso position
        torso_height = np.array(p_torso[2]).reshape((-1))
        # Torso y value
        torso_y_value = np.array(p_torso[1]).reshape((-1))
        # Contact information
        contact_info = np.zeros(self.env.n_sensor)
        contact_idxs = np.where(self.env.get_sensor_values(sensor_names=self.env.sensor_names) > 0.2)[0]
        contact_info[contact_idxs] = 1.0 # 1 means contact occurred
        # Concatenate information
        state = np.concatenate([
            qpos,
            qvel/10.0, # scale
            R_torso_flat,
            torso_height,
            torso_y_value,
            contact_info
        ])
        return state
    
    def get_observation(self):
        """
            Get observation 
        """
        
        # Sparsely accumulated history vector 
        state_history_sparse = self.state_history[self.history_ticks,:]
        
        # Concatenate information
        obs = np.concatenate([
            state_history_sparse
        ])
        return obs.flatten()

    def sample_action(self):
        """
            Sample action (8)
        """
        a_min  = self.env.ctrl_ranges[:,0]
        a_max  = self.env.ctrl_ranges[:,1]
        action = a_min + (a_max-a_min)*np.random.rand(len(a_min))
        return action
        
    def step(self,a,max_time=np.inf):
        # ── 0) 공통: 이전 상태 저장 ──────────────────────────────────
        self.tick += 1
        p_prev = self.env.get_p_body('torso')

        # ── 1) 시뮬레이션 실행 ─────────────────────────────────────
        self.env.step(ctrl=a, nstep=self.mujoco_nstep)

        # ── 2) 현재 상태 ──────────────────────────────────────────
        p_cur = self.env.get_p_body('torso')
        R_cur = self.env.get_R_body('torso')

        # ========== 3. 보상 항목 수정 ==========
        # 3-1) ‘수직 상승’ 보상  (기존 r_forward → r_vertical)
        z_diff      = p_cur[2] - p_prev[2]               # ↑ 방향 변위
        r_vertical  = 5 * z_diff / self.dt if z_diff > 0 else z_diff / self.dt                  # 순간 상승 속도

        # 3-2) 힙 고정 패널티  (각도 제곱합/속도 제곱합 둘 중 하나 선택)
        hip_idx  = [0, 2, 4, 6]                          # ctrl_qpos 순서
        qpos     = self.env.data.qpos[self.env.ctrl_qpos_idxs][hip_idx]
        qvel     = self.env.data.qvel[self.env.ctrl_qvel_idxs][hip_idx]
        # 각도 기준: 너무 벌어지면 패
        hip_pen_ang = -0.2 * np.sum(qpos**2)             # 계수는 임의–조절
        # 속도 기준: 꿈틀거리면 패
        hip_pen_vel = -0.1 * np.sum((qvel/10.0)**2)

        knee_idx = [1, 3, 5, 7]
        qpos_knee = self.env.data.qpos[self.env.ctrl_qpos_idxs][knee_idx]
        qvel_knee = self.env.data.qvel[self.env.ctrl_qpos_idxs][knee_idx]

        knee_bon_ang = 0.4 * np.sum(qpos_knee**2)
        knee_bon_vel = 0.4 * np.sum((qvel_knee/10.0)**2)

        # 3-3) 기존 항목에서 forward / heading / lane 제거
        #     → r = r_vertical + …만 사용
        ROLLOVER = (np.dot(R_cur[:,2],np.array([0,0,1]))<0.0)
        if (self.get_sim_time() >= max_time) or ROLLOVER:
            d = True
        else:
            d = False
        r_survive = -10.0 if ROLLOVER else 0.01

        r = r_vertical + knee_bon_ang + knee_bon_vel + r_survive + hip_pen_ang + hip_pen_vel
        r = np.array(r)
        
        # Accumulate state history (update 'state_history')
        self.accumulate_state_history()
        
        # Next observation 'accumulate_state_history' should be called before calling 'get_observation'
        o_prime = self.get_observation()
        
        # Other information
        info = {
            # 높이·속도
            'h_prev'      : p_prev[2],
            'h_cur'       : p_cur[2],
            'z_diff'      : z_diff,          # 이번 스텝에서 상승한 거리
            'v_z'         : r_vertical,      # 순간 수직 속도 (= 보상 전용)

            # 힙 고정 패널티 항목
            'hip_pen_ang' : hip_pen_ang,
            'hip_pen_vel' : hip_pen_vel,

            # 생존 여부
            'rollover'    : ROLLOVER,
            'r_survive'   : r_survive,

            # 최종 스텝별 보상
            'reward_step' : r,
        }

        
        # Return
        return o_prime,r,d,info
    
    def render(
            self,
            TRACK_TORSO      = True,
            PLOT_WORLD_COORD = True,
            PLOT_TORSO_COORD = True,
            PLOT_SENSOR      = True,
            PLOT_CONTACT     = True,
            PLOT_TIME        = True,
        ):
        """
            Render
        """
        # Change lookat
        if TRACK_TORSO:
            p_lookat = self.env.get_p_body('torso')
            self.env.set_viewer(lookat=p_lookat)
        # World coordinate
        if PLOT_WORLD_COORD:
            self.env.plot_T(p=np.zeros(3),R=np.eye(3,3),plot_axis=True,axis_len=1.0,axis_width=0.0025)
        # Plot snapbot base
        if PLOT_TORSO_COORD:
            p_torso,R_torso = self.env.get_pR_body(body_name='torso') # update viewer
            self.env.plot_T(p=p_torso,R=R_torso,plot_axis=True,axis_len=0.25,axis_width=0.0025)
        # Plot contact sensors
        if PLOT_SENSOR:
            contact_idxs = np.where(self.env.get_sensor_values(sensor_names=self.env.sensor_names) > 0.2)[0]
            for idx in contact_idxs:
                sensor_name = self.env.sensor_names[idx]
                p_sensor = self.env.get_p_sensor(sensor_name)
                self.env.plot_sphere(p=p_sensor,r=0.02,rgba=[1,0,0,0.2])
        # Plot contact info
        if PLOT_CONTACT:
            self.env.plot_contact_info()
        # Plot time and tick on top of torso
        if PLOT_TIME:
            self.env.plot_T(p=p_torso+0.25*R_torso[:,2],R=np.eye(3,3),
                       plot_axis=False,label='[%.2f]sec'%(self.env.get_sim_time()))
        # Do render
        self.env.render()

    def reset(self):
        """
            Reset
        """
        # Reset parameters
        self.tick = 0
        # Reset env
        self.env.reset(step=True)
        # Reset history
        self.state_history = np.zeros((self.n_history,self.state_dim))
        self.tick_history  = np.zeros((self.n_history,1))
        # Get observation
        o = self.get_observation()
        return o
        
    def init_viewer(self):
        """
            Initialize viewer
        """
        self.env.init_viewer(distance=3.0)
        
    def close_viewer(self):
        """
            Close viewer
        """
        self.env.close_viewer()
        
    def get_sim_time(self):
        """
            Get time (sec)
        """
        return self.env.get_sim_time()
    
    def is_viewer_alive(self):
        """
            Check whether the viewer is alive
        """
        return self.env.is_viewer_alive()
    
    def accumulate_state_history(self):
        """
            Get state history
        """
        state = self.get_state()
        # Accumulate 'state' and 'tick'
        self.state_history[1:,:] = self.state_history[:-1,:]
        self.state_history[0,:]  = state
        self.tick_history[1:,:]  = self.tick_history[:-1,:]
        self.tick_history[0,:]   = self.tick
        
    def viewer_pause(self):
        """
            Viewer pause
        """
        self.env.viewer_pause()
        
    def grab_image(self,resize_rate=1.0,interpolation=0):
        """
            Grab image
        """
        return self.env.grab_image(rsz_rate=resize_rate,interpolation=interpolation)