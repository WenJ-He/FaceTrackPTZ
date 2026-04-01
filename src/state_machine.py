"""Explicit state machine for FaceTrack PTZ.

States and transitions per 14.9:

  INIT -> CONNECTING_STREAM
  CONNECTING_STREAM -> DETECTING
  DETECTING -> SORTING | DETECTING (no face) | ERROR
  SORTING -> MOVING | DETECTING (no valid target) | ERROR
  MOVING -> RECOGNIZING | NEXT_TARGET (ptz failed) | ERROR
  RECOGNIZING -> HOLDING | MOVING (more stages) | ERROR
  HOLDING -> NEXT_TARGET | ERROR
  NEXT_TARGET -> DETECTING | ERROR
  ERROR -> RECOVER
  RECOVER -> DETECTING
  * -> ERROR (any state can transition to ERROR)
"""

from __future__ import annotations

import enum
import threading
from typing import Optional


class State(enum.Enum):
    INIT = "INIT"
    CONNECTING_STREAM = "CONNECTING_STREAM"
    DETECTING = "DETECTING"
    SORTING = "SORTING"
    MOVING = "MOVING"
    RECOGNIZING = "RECOGNIZING"
    HOLDING = "HOLDING"
    NEXT_TARGET = "NEXT_TARGET"
    ERROR = "ERROR"
    RECOVER = "RECOVER"


ALLOWED_TRANSITIONS = {
    State.INIT: {State.CONNECTING_STREAM, State.ERROR},
    State.CONNECTING_STREAM: {State.DETECTING, State.ERROR},
    State.DETECTING: {State.SORTING, State.DETECTING, State.ERROR},
    State.SORTING: {State.MOVING, State.DETECTING, State.ERROR},
    State.MOVING: {State.RECOGNIZING, State.NEXT_TARGET, State.ERROR},
    State.RECOGNIZING: {State.HOLDING, State.MOVING, State.ERROR},
    State.HOLDING: {State.NEXT_TARGET, State.ERROR},
    State.NEXT_TARGET: {State.DETECTING, State.ERROR},
    State.ERROR: {State.RECOVER},
    State.RECOVER: {State.DETECTING},
}


class StateMachine:
    """Thread-safe state machine with transition logging."""

    def __init__(self) -> None:
        self._state = State.INIT
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def transition(self, new_state: State) -> None:
        with self._lock:
            if new_state not in ALLOWED_TRANSITIONS.get(self._state, set()):
                from . import logger as log
                log.log(
                    f"Invalid transition: {self._state.value} -> {new_state.value}",
                    state=self._state.value,
                    result="INVALID_TRANSITION",
                )
                return
            old = self._state
            self._state = new_state
        from . import logger as log
        log.log(
            f"State: {old.value} -> {new_state.value}",
            state=new_state.value,
            result="TRANSITION_OK",
        )

    def force_state(self, new_state: State) -> None:
        with self._lock:
            self._state = new_state
