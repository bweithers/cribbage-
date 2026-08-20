"""A two-player cribbage simulator, built as the foundation for a search/NN engine.

The layers are independent and testable bottom-up:

* :mod:`cribbage.cards` -- cards as plain ints.
* :mod:`cribbage.scoring` -- table-driven scoring for the show and the play.
* :mod:`cribbage.engine` -- the state machine, including ``clone`` and
  ``determinize`` for imperfect-information search.
* :mod:`cribbage.agents` -- the agent interface and baseline opponents.
* :mod:`cribbage.arena` -- match running and statistics.
"""

from .cards import card_str, hand_str, parse_card, parse_hand
from .engine import CribbageState, GameResult, InfoState, Phase, new_game
from .scoring import score_hand, score_hand_detail, score_play, score_play_detail

__version__ = "0.1.0"

__all__ = [
    "CribbageState",
    "GameResult",
    "InfoState",
    "Phase",
    "new_game",
    "score_hand",
    "score_hand_detail",
    "score_play",
    "score_play_detail",
    "card_str",
    "hand_str",
    "parse_card",
    "parse_hand",
    "__version__",
]
