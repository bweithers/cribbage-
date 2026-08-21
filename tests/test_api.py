"""The HTTP layer: stateless replay, validation, and information hygiene.

The web client is a drawing surface with no rules in it, so everything worth
testing about it is on this side: that a game rebuilt from its action list is
the same game, that a bad action is refused rather than half-applied, and above
all that the JSON never carries a card the player is not entitled to see.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="the web extra is not installed")
from fastapi.testclient import TestClient  # noqa: E402

from cribbage.api import MAX_ACTIONS, GameConfig, app, replay  # noqa: E402
from cribbage.engine import Phase  # noqa: E402

client = TestClient(app)


def start(**options) -> dict:
    response = client.post("/api/new", json=options)
    assert response.status_code == 200, response.text
    return response.json()


def advance(config, actions: list) -> dict:
    if isinstance(config, GameConfig):
        config = config.model_dump()
    response = client.post("/api/state", json={"config": config, "actions": actions})
    assert response.status_code == 200, response.text
    return response.json()


def all_card_ids(node) -> set[int]:
    """Every card id anywhere in a view, however deeply nested."""
    found: set[int] = set()
    if isinstance(node, dict):
        if "id" in node and "suit" in node and "rank" in node:
            found.add(node["id"])
        for value in node.values():
            found |= all_card_ids(value)
    elif isinstance(node, list):
        for value in node:
            found |= all_card_ids(value)
    return found


# ----------------------------------------------------------------------
# Starting and replaying
# ----------------------------------------------------------------------


def test_new_game_fills_in_a_seed_and_deals():
    data = start(luck=75)
    assert isinstance(data["config"]["seed"], int)
    view = data["view"]
    assert view["phase"] == "discard"
    assert view["yourTurn"] is True
    assert len(view["hand"]) == 6
    assert len(view["legal"]) == 15
    assert view["starter"] is None


def test_replay_is_deterministic():
    """The whole design rests on this: same inputs, same game, every time."""
    data = start(seed=11, luck=50)
    config, actions = data["config"], [data["view"]["legal"][3]]
    first = advance(config, actions)["view"]
    second = advance(config, actions)["view"]
    assert first == second


def test_the_action_list_is_the_only_state():
    """Dropping the last action must rewind exactly -- that is what undo is."""
    data = start(seed=5)
    config = data["config"]
    one = [data["view"]["legal"][0]]
    after_one = advance(config, one)["view"]

    play = after_one["legal"][0]
    after_two = advance(config, one + [play])["view"]
    assert after_two != after_one

    rewound = advance(config, one)["view"]
    assert rewound == after_one


def test_the_opponents_moves_are_recomputed_not_supplied():
    """A client sends only its own decisions, so it cannot forge the opponent's."""
    data = start(seed=9)
    config = GameConfig(**data["config"])
    state = replay(config, [data["view"]["legal"][0]])
    # Both sides have acted, and the cut has happened, off one human action.
    assert state.phase is Phase.PLAY
    assert state.starter is not None


def test_a_whole_game_can_be_played_through_the_api():
    data = start(seed=21)
    config, actions = data["config"], []
    view = data["view"]
    for _ in range(400):
        if view["phase"] == "over":
            break
        actions.append(view["legal"][0])
        view = advance(config, actions)["view"]
    assert view["phase"] == "over"
    assert view["result"] is not None
    assert view["cutLuck"] is not None
    assert view["result"]["scores"]["you"] >= 121 or (
        view["result"]["scores"]["opponent"] >= 121
    )


def test_rigging_the_cut_reaches_the_engine():
    best = start(seed=11, luck=100)
    worst = start(seed=11, luck=0)
    lay = best["view"]["legal"][0]
    top = advance(best["config"], [lay])["view"]["starter"]["id"]
    bottom = advance(worst["config"], [lay])["view"]["starter"]["id"]
    assert top != bottom


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_an_illegal_action_is_refused_with_a_reason():
    data = start(seed=11)
    held = {card["id"] for card in data["view"]["hand"]}
    outsider = next(card for card in range(52) if card not in held)
    response = client.post(
        "/api/state",
        json={"config": data["config"], "actions": [[outsider, outsider + 1]]},
    )
    assert response.status_code == 400
    assert "cannot discard" in response.json()["detail"]


def test_a_lay_away_of_the_wrong_size_is_refused():
    data = start(seed=11)
    card = data["view"]["hand"][0]["id"]
    response = client.post(
        "/api/state", json={"config": data["config"], "actions": [[card]]}
    )
    assert response.status_code == 400
    assert "two cards" in response.json()["detail"]


def test_extra_actions_are_refused_rather_than_ignored():
    data = start(seed=11)
    config = data["config"]
    actions = []
    view = data["view"]
    for _ in range(400):
        if view["phase"] == "over":
            break
        actions.append(view["legal"][0])
        view = advance(config, actions)["view"]

    response = client.post(
        "/api/state", json={"config": config, "actions": actions + [[0]]}
    )
    assert response.status_code == 400
    assert "left over" in response.json()["detail"]


def test_an_unknown_opponent_is_rejected():
    assert client.post("/api/new", json={"opponent": "nonesuch"}).status_code == 422


def test_absurd_settings_are_rejected():
    assert client.post("/api/new", json={"luck": 500}).status_code == 422
    assert client.post("/api/new", json={"target": 0}).status_code == 422
    assert client.post(
        "/api/state",
        json={"config": {"seed": 1}, "actions": [[0, 1]] * (MAX_ACTIONS + 1)},
    ).status_code == 422


# ----------------------------------------------------------------------
# Information hygiene
# ----------------------------------------------------------------------


def test_the_view_never_carries_a_card_the_player_cannot_see():
    """Checked against the engine's own hidden state, at every decision."""
    data = start(seed=13)
    config = GameConfig(**data["config"])
    actions: list = []
    view = data["view"]
    checked = 0

    for _ in range(400):
        if view["phase"] == "over":
            break
        state = replay(config, actions)
        opponent = 1 - config.human
        hidden = (
            set(state.hands[opponent]) | set(state.discards[opponent])
            | set(state.deck)
        )
        shown = all_card_ids(view)
        assert shown.isdisjoint(hidden), (
            f"view exposed {sorted(shown & hidden)} at round {view['round']}"
        )
        checked += 1
        actions.append(view["legal"][0])
        view = advance(config, actions)["view"]

    assert checked > 20, f"only checked {checked} decisions"


def test_the_opening_view_says_nothing_about_the_opponents_deal():
    data = start(seed=13)
    config = GameConfig(**data["config"])
    state = replay(config, [])
    theirs = set(state.hands[1 - config.human])
    assert all_card_ids(data["view"]).isdisjoint(theirs)
    assert not any(event["cards"] for event in data["view"]["events"]
                   if event["actor"] == "opponent")


# ----------------------------------------------------------------------
# Serving
# ----------------------------------------------------------------------


def test_the_page_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>Cribbage</title>" in response.text


def test_the_agent_list_is_offered():
    agents = client.get("/api/agents").json()["agents"]
    assert "heuristic" in agents and "random" in agents
