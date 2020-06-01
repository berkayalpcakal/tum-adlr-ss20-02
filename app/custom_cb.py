from stable_baselines.common.callbacks import BaseCallback


class ResetEnvCb(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_rollout_end(self) -> None:
        self.model.get_env().reset()
