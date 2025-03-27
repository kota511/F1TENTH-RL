from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class SurvivalEvalCallback(BaseCallback):
    def __init__(self, eval_env, check_freq=5000, target_survival_steps=2000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.target_survival_steps = target_survival_steps

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            obs = self.eval_env.reset()
            done = False
            steps_survived = 0
            max_speed = 0.0
            max_angle_error = 0.0
            last_observations = []

            while not done and steps_survived < self.target_survival_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                steps_survived += 1

                # Debugging values from the raw observation
                raw_obs = self.eval_env.envs[0].obs
                speed = raw_obs['linear_vels_x'][0]
                angle_err = self.eval_env.envs[0].angle_error

                max_speed = max(max_speed, speed)
                max_angle_error = max(max_angle_error, abs(angle_err))

                if steps_survived > self.target_survival_steps - 50:
                    last_observations.append({
                        "step": steps_survived,
                        "speed": speed,
                        "angle_error": angle_err,
                        "pose_x": raw_obs['poses_x'][0],
                        "pose_y": raw_obs['poses_y'][0],
                        "theta": raw_obs['poses_theta'][0]
                    })

            if self.verbose > 0:
                print(f"[Eval] Survived {steps_survived} steps")
                print(f"[Eval] Max speed: {max_speed:.2f} m/s")
                print(f"[Eval] Max angle error: {max_angle_error:.4f} rad")

            if steps_survived < self.target_survival_steps:
                print("[Eval] Final few observations before crash or timeout:")
                for o in last_observations[-10:]:
                    print(f"  Step {o['step']} | Speed: {o['speed']:.2f}, AngleErr: {o['angle_error']:.3f}, Pos: ({o['pose_x']:.2f}, {o['pose_y']:.2f}), Theta: {o['theta']:.2f}")

            if steps_survived >= self.target_survival_steps:
                print(f"✅ Agent survived {steps_survived} steps (~{steps_survived * 0.01:.2f}s). Stopping training.")
                return False  # Stop training

        return True
