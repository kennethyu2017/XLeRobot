from dataclasses import dataclass
import draccus

from lerobot.robots import (  # noqa: F401
    RobotConfig,
    # bi_so_follower,
    # koch_follower,
    # lekiwi,
    # make_robot_from_config,
    # omx_follower,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    # bi_so_leader,
    # koch_leader,
    # make_teleoperator_from_config,
    # omx_leader,
    so_leader,
)

@dataclass
class MockConfig:
	teleop: TeleoperatorConfig | None = None
	robot: RobotConfig | None = None

	def __str__(self):
		return f"MockConfig: {self.teleop=:}, {self.robot=:}"


@draccus.wrap()
def print_cfg(cfg: MockConfig):
	print(cfg)

def main():
	print_cfg()

if __name__ == "__main__":
	main()

