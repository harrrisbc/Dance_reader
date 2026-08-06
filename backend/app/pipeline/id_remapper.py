"""Map ByteTrack raw ids to compact reusable slot ids (0, 1, 2, ...).

Adapted from pose_bridge IdSlotRemapper for offline dance analysis.
Pass video time as `now` so grace windows are in seconds of footage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class _GraceSlot:
    slot: int
    com: Optional[Tuple[float, float]]
    freed_at: float


@dataclass
class IdSlotRemapper:
    """
    Remap tracker raw ids → compact display slots.

    - Active people get the lowest free slots 0..max_slots-1
    - When a person disappears, their slot enters grace (COM reclaim)
    - Lock mode: only reclaim locked slots; new strangers cannot steal them
    """

    grace_sec: float = 2.5
    com_match_dist: float = 0.22
    max_slots: int = 2

    _raw_to_slot: Dict[int, int] = field(default_factory=dict, init=False)
    _slot_to_raw: Dict[int, int] = field(default_factory=dict, init=False)
    _slot_com: Dict[int, Optional[Tuple[float, float]]] = field(default_factory=dict, init=False)
    _grace: List[_GraceSlot] = field(default_factory=list, init=False)
    _locked: bool = field(default=False, init=False)
    _locked_slots: Set[int] = field(default_factory=set, init=False)

    def reset(self) -> None:
        self._raw_to_slot.clear()
        self._slot_to_raw.clear()
        self._slot_com.clear()
        self._grace.clear()
        self._locked = False
        self._locked_slots.clear()

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def locked_slots(self) -> Set[int]:
        return set(self._locked_slots)

    def set_grace_sec(self, sec: float) -> None:
        self.grace_sec = max(0.1, float(sec))

    def set_max_slots(self, max_slots: int) -> None:
        self.max_slots = max(1, int(max_slots))
        for raw, slot in list(self._raw_to_slot.items()):
            if slot >= self.max_slots:
                self._raw_to_slot.pop(raw, None)
                self._slot_to_raw.pop(slot, None)
                self._slot_com.pop(slot, None)
        self._grace = [g for g in self._grace if g.slot < self.max_slots]
        self._locked_slots = {s for s in self._locked_slots if s < self.max_slots}

    def lock_current(self) -> Set[int]:
        occupied = set(self._slot_to_raw.keys())
        if not occupied:
            self._locked = False
            self._locked_slots.clear()
            return set()
        self._locked = True
        self._locked_slots = occupied
        return set(self._locked_slots)

    def unlock(self) -> None:
        self._locked = False
        self._locked_slots.clear()

    def _lowest_free_slot(self) -> Optional[int]:
        used = set(self._slot_to_raw.keys())
        for slot in range(self.max_slots):
            if slot not in used:
                return slot
        return None

    def _expire_grace(self, now: float) -> None:
        hold = self.grace_sec
        if self._locked:
            hold = max(hold, self.grace_sec * 2.0)
        self._grace = [
            g
            for g in self._grace
            if (now - g.freed_at) <= hold and g.slot < self.max_slots
        ]

    def _try_reclaim_grace(
        self, com: Optional[Tuple[float, float]], now: float
    ) -> Optional[int]:
        self._expire_grace(now)
        if com is None or not self._grace:
            return None

        best_i = -1
        best_d = self.com_match_dist
        if self._locked:
            best_d = max(best_d, 0.28)

        for i, g in enumerate(self._grace):
            if g.com is None or g.slot in self._slot_to_raw:
                continue
            if self._locked and g.slot not in self._locked_slots:
                continue
            d = ((com[0] - g.com[0]) ** 2 + (com[1] - g.com[1]) ** 2) ** 0.5
            if d <= best_d:
                best_d = d
                best_i = i
        if best_i < 0:
            return None
        return self._grace.pop(best_i).slot

    def update(
        self,
        detections: List[Tuple[int, Optional[Tuple[float, float]]]],
        now: Optional[float] = None,
    ) -> Dict[int, int]:
        import time

        now = time.perf_counter() if now is None else now
        self._expire_grace(now)

        active_raw = {raw for raw, _ in detections}
        com_by_raw = {raw: com for raw, com in detections}

        for raw in list(self._raw_to_slot.keys()):
            if raw not in active_raw:
                slot = self._raw_to_slot.pop(raw)
                self._slot_to_raw.pop(slot, None)
                last_com = self._slot_com.pop(slot, None)
                self._grace.append(_GraceSlot(slot=slot, com=last_com, freed_at=now))

        for raw, com in detections:
            if raw in self._raw_to_slot:
                slot = self._raw_to_slot[raw]
                if slot >= self.max_slots:
                    self._raw_to_slot.pop(raw, None)
                    self._slot_to_raw.pop(slot, None)
                    self._slot_com.pop(slot, None)
                else:
                    if com is not None:
                        self._slot_com[slot] = com
                    continue

            reclaimed = self._try_reclaim_grace(com, now)
            if reclaimed is not None:
                slot = reclaimed
            elif self._locked:
                continue
            else:
                slot = self._lowest_free_slot()

            if slot is None:
                continue
            if slot in self._slot_to_raw:
                slot = self._lowest_free_slot()
            if slot is None:
                continue

            self._raw_to_slot[raw] = slot
            self._slot_to_raw[slot] = raw
            self._slot_com[slot] = com if com is not None else com_by_raw.get(raw)

        return dict(self._raw_to_slot)
