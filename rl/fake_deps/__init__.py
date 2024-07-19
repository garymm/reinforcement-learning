"""A module that inserts fakes for dependencies that acme imports but
doesn't actually use.
"""

import sys

from rl.fake_deps import gym, sonnet

sys.modules["gym"] = gym
sys.modules["sonnet"] = sonnet
