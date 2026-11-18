"""Wave Function Collapse, ported from Ggargur/WFC (Unity/C#) to plain Python.

Same two phases as the original package:

* ``learn``  (WFCGenerator): scan a hand-made sample map, record which tile
  can sit next to which in every direction, how often each tile appears
  (its weight) and which tiles may touch the grid border.
* ``MapCreator``: fill a grid chunk by chunk. Inside a chunk, pick the
  uncollapsed cell with the lowest Shannon entropy, collapse it by weighted
  random choice, and backtrack when a neighbour is left with no options.
  A chunk that keeps failing is reset and retried; after ``max_chunk_retries``
  the seed is declared unfortunate.

ASCII tiles never rotate, so the rotation logic of the original
(``MaxTurns``) is dropped.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# Direction -> (dx, dy). y grows downwards, like text.
DIRECTIONS = {"up": (0, -1), "right": (1, 0), "down": (0, 1), "left": (-1, 0)}


class UnfortunateSeedError(Exception):
    """A chunk could not be solved within the retry budget."""


@dataclass
class CellType:
    tile: str
    weight: int = 0
    compatible_with_edges: bool = False
    neighbors: dict[str, set[str]] = field(default_factory=lambda: {d: set() for d in DIRECTIONS})

    def is_compatible(self, direction: str, other: str) -> bool:
        return other in self.neighbors[direction]


def learn(sample: list[str]) -> dict[str, CellType]:
    """Build the tile database from a sample map (WFCGenerator)."""
    height = len(sample)
    width = max(len(row) for row in sample)
    grid = [row.ljust(width) for row in sample]
    types: dict[str, CellType] = {}
    for y in range(height):
        for x in range(width):
            tile = grid[y][x]
            cell_type = types.setdefault(tile, CellType(tile))
            cell_type.weight += 1
            for direction, (dx, dy) in DIRECTIONS.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    cell_type.neighbors[direction].add(grid[ny][nx])
                else:
                    cell_type.compatible_with_edges = True
    return types


class Cell:
    def __init__(self, x: int, y: int, creator: "MapCreator"):
        self.x, self.y = x, y
        self.creator = creator
        self.collapsed: str | None = None
        # Options already tried under a given set of collapsed neighbours.
        self.tested: dict[tuple, set[str]] = {}
        self._possibilities: list[str] | None = None
        self.entropy = math.inf

    @property
    def is_collapsed(self) -> bool:
        return self.collapsed is not None

    @property
    def is_collapsable(self) -> bool:
        return not self.is_collapsed and bool(self.possibilities)

    def snapshot(self) -> tuple:
        return tuple(n.collapsed if n else None for n in self.creator.neighbors(self).values())

    def set_dirty(self) -> None:
        self._possibilities = None

    @property
    def possibilities(self) -> list[str]:
        if self._possibilities is not None:
            return self._possibilities
        types = self.creator.types
        tried = self.tested.get(self.snapshot(), set())
        result = []
        for tile, cell_type in types.items():
            if tile in tried:
                continue
            valid = True
            for direction, neighbor in self.creator.neighbors(self).items():
                if neighbor is None:
                    if not cell_type.compatible_with_edges:
                        valid = False
                        break
                    continue
                if neighbor.collapsed is None:
                    continue
                if not cell_type.is_compatible(direction, neighbor.collapsed):
                    valid = False
                    break
            if valid:
                result.append(tile)
        self._possibilities = result
        weights = [types[t].weight for t in result]
        total = sum(weights)
        self.entropy = (
            -sum((w / total) * math.log2(w / total) for w in weights if w) if total else math.inf
        )
        return result

    def collapse(self) -> bool:
        snapshot = self.snapshot()
        options = list(self.possibilities)
        rng = self.creator.rng
        types = self.creator.types
        if not options:
            return False
        choice = rng.choices(options, weights=[types[t].weight for t in options])[0]
        self.tested.setdefault(snapshot, set()).add(choice)
        self.collapsed = choice
        self.creator.on_collapsed(self)
        return True

    def uncollapse(self) -> None:
        self.collapsed = None
        self.set_dirty()
        self.creator.on_collapsed(self)

    def reset(self) -> None:
        self.collapsed = None
        self.tested.clear()
        self.set_dirty()


class MapCreator:
    def __init__(
        self,
        types: dict[str, CellType],
        width: int,
        height: int,
        seed: int,
        chunk_size: tuple[int, int] = (13, 7),
        max_chunk_retries: int = 10,
        max_iterations: int | None = None,
    ):
        self.types = types
        self.width, self.height = width, height
        self.rng = random.Random(seed)
        self.chunk_size = chunk_size
        self.max_chunk_retries = max_chunk_retries
        self.max_iterations = max_iterations or width * height * 4
        self.grid = [[Cell(x, y, self) for x in range(width)] for y in range(height)]
        self._neighbors: dict[tuple[int, int], dict[str, Cell | None]] = {}
        self.backtracks = 0
        self.chunk_restarts = 0

    def neighbors(self, cell: Cell) -> dict[str, Cell | None]:
        key = (cell.x, cell.y)
        if key not in self._neighbors:
            found = {}
            for direction, (dx, dy) in DIRECTIONS.items():
                nx, ny = cell.x + dx, cell.y + dy
                inside = 0 <= nx < self.width and 0 <= ny < self.height
                found[direction] = self.grid[ny][nx] if inside else None
            self._neighbors[key] = found
        return self._neighbors[key]

    def on_collapsed(self, cell: Cell) -> None:
        for neighbor in self.neighbors(cell).values():
            if neighbor is not None:
                neighbor.set_dirty()

    def chunks(self):
        cw, ch = self.chunk_size
        for cy in range(0, self.height, ch):
            for cx in range(0, self.width, cw):
                yield [
                    self.grid[y][x]
                    for y in range(cy, min(cy + ch, self.height))
                    for x in range(cx, min(cx + cw, self.width))
                ]

    def select_next(self, chunk: list[Cell]) -> Cell | None:
        options = [c for c in chunk if c.is_collapsable]
        if not options:
            return None
        lowest = min(c.entropy for c in options)
        return self.rng.choice([c for c in options if math.isclose(c.entropy, lowest)])

    def neighbors_have_mistake(self, cell: Cell) -> bool:
        return any(
            n is not None and not n.is_collapsed and not n.is_collapsable
            for n in self.neighbors(cell).values()
        )

    def collapse_in_chunk(self, nxt: Cell | None, stack: list[Cell]) -> bool:
        while True:
            if nxt is None:
                return False
            if nxt.collapse():
                stack.append(nxt)
                if not self.neighbors_have_mistake(nxt):
                    return True
            if not stack:
                return False
            # Backtrack inside the chunk only.
            nxt = stack.pop()
            self.backtracks += 1
            nxt.uncollapse()

    def solve_chunk(self, chunk: list[Cell]) -> None:
        for _ in range(self.max_chunk_retries):
            stack: list[Cell] = []
            failed = False
            iteration = 0
            while any(not c.is_collapsed for c in chunk):
                iteration += 1
                if iteration > self.max_iterations:
                    failed = True
                    break
                if not self.collapse_in_chunk(self.select_next(chunk), stack):
                    failed = True
                    break
            if not failed:
                return
            self.chunk_restarts += 1
            for cell in chunk:
                cell.reset()
            for cell in chunk:
                self.on_collapsed(cell)
        raise UnfortunateSeedError

    def create(self) -> list[str]:
        for chunk in self.chunks():
            self.solve_chunk(chunk)
        return ["".join(c.collapsed for c in row) for row in self.grid]
