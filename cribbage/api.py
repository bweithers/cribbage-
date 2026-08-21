"""A FastAPI front door to the engine, so a browser can play without a rules port.

**No session state.**  A cribbage game is a pure function of its configuration
and the human's decisions: the deal comes from a seed, the opponent is
deterministic, and a rigged cut is decided from the position rather than from
chance.  So the client holds ``{config, actions}`` and the server replays the
whole game on every request.  A replay costs well under a millisecond against an
engine that does ~80k ``apply_action`` per second, and in exchange there is no
session store to expire, leak, or lose across a restart -- and undo is just
dropping the last action.

**No rules in the browser.**  Every legality decision, every point scored and
every redaction happens here, against the same tested engine the analysis
scripts use.  The page is presentation only.

Run it with::

    pip install 'cribbage[web]'
    uvicorn cribbage.api:app --reload

then open http://127.0.0.1:8000.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from .agents import AGENTS, make_agent
from .engine import DEFAULT_TARGET, CribbageState, Phase
from .luck import PercentileCut
from .view import game_view

__all__ = ["app", "GameConfig", "StateRequest", "replay"]

#: A game cannot need more decisions than this; the cap keeps a malicious or
#: buggy client from asking the server to replay forever.
MAX_ACTIONS = 400

WEB_ROOT = Path(__file__).parent / "web"

app = FastAPI(
    title="cribbage",
    description="A two-player cribbage engine, exposed for a browser client.",
    version="0.1.0",
)


class GameConfig(BaseModel):
    """Everything needed to reproduce a game exactly."""

    seed: int = Field(..., ge=0, lt=1 << 62)
    dealer: int = Field(0, ge=0, le=1)
    human: int = Field(0, ge=0, le=1)
    target: int = Field(DEFAULT_TARGET, ge=2, le=1000)
    luck: Optional[float] = Field(
        None, ge=0, le=100,
        description="Rig every cut to this percentile of what it could have been "
                    "for the human. Omit for an honest game.",
    )
    opponent: str = "heuristic"

    @field_validator("opponent")
    @classmethod
    def _known_agent(cls, value: str) -> str:
        if value not in AGENTS:
            raise ValueError(f"unknown agent {value!r}; try one of {sorted(AGENTS)}")
        return value


class NewGameRequest(BaseModel):
    seed: Optional[int] = Field(None, ge=0, lt=1 << 62)
    dealer: int = Field(0, ge=0, le=1)
    target: int = Field(DEFAULT_TARGET, ge=2, le=1000)
    luck: Optional[float] = Field(None, ge=0, le=100)
    opponent: str = "heuristic"

    # Without this the bad name slips through request validation and only fails
    # when GameConfig is built inside the handler -- a 500 where a 422 belongs.
    _known_agent = field_validator("opponent")(GameConfig._known_agent.__func__)


class StateRequest(BaseModel):
    config: GameConfig
    actions: list[list[int]] = Field(default_factory=list, max_length=MAX_ACTIONS)


def _as_action(raw: list[int], state: CribbageState):
    """Turn a client's list of card ints into an engine action."""
    if state.phase is Phase.DISCARD:
        if len(raw) != 2:
            raise ValueError(f"a lay-away needs two cards, got {len(raw)}")
        return tuple(sorted(raw))
    if len(raw) != 1:
        raise ValueError(f"a play is one card, got {len(raw)}")
    return raw[0]


def replay(config: GameConfig, actions: list[list[int]]) -> CribbageState:
    """Rebuild a game from its configuration and the human's decisions.

    The opponent's moves are recomputed rather than sent by the client, so a
    client can neither see them in advance nor forge them.  Returns the state
    sitting on the human's next decision, or terminal.
    """
    chooser = (
        None if config.luck is None else PercentileCut(config.human, config.luck)
    )
    state = CribbageState(
        dealer=config.dealer, target=config.target,
        seed=config.seed, cut_chooser=chooser,
    )
    opponent = make_agent(config.opponent, seed=config.seed)

    pending = list(actions)
    while not state.is_terminal():
        player = state.current_player
        assert player is not None
        if player == config.human:
            if not pending:
                break
            state.apply_action(_as_action(pending.pop(0), state))
        else:
            state.apply_action(opponent.act(state.information_state(player)))

    if pending:
        raise ValueError(
            f"{len(pending)} action(s) left over: the game did not need them"
        )
    return state


def _view_or_400(config: GameConfig, actions: list[list[int]]) -> dict:
    try:
        state = replay(config, actions)
    except ValueError as problem:
        raise HTTPException(status_code=400, detail=str(problem)) from None
    return {"config": config.model_dump(), "actions": actions,
            "view": game_view(state, config.human)}


@app.post("/api/new", summary="Start a game")
def new_game(request: NewGameRequest) -> dict:
    """Fill in a seed if the client did not choose one, and deal."""
    config = GameConfig(
        seed=random.randrange(1 << 30) if request.seed is None else request.seed,
        dealer=request.dealer, human=0, target=request.target,
        luck=request.luck, opponent=request.opponent,
    )
    return _view_or_400(config, [])


@app.post("/api/state", summary="Replay a game and return the player's view")
def state(request: StateRequest) -> dict:
    """The only endpoint the client needs after starting: post the whole action
    list, get back what the player is allowed to see."""
    return _view_or_400(request.config, request.actions)


@app.get("/api/agents", summary="Opponents this build can offer")
def agents() -> dict:
    return {"agents": sorted(AGENTS)}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    page = WEB_ROOT / "index.html"
    if not page.exists():  # pragma: no cover - only if the package is trimmed
        raise HTTPException(status_code=404, detail="the web client is not installed")
    return FileResponse(page)
