"""Generate the daily dungeon map for the profile README.

Usage:
    python3 dungeon.py --out dist [--seed 2026-11-18] [--offline]

Reads GITHUB_TOKEN from the environment to fetch profile stats. Writes
dungeon-dark.svg and dungeon-light.svg to the output directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import urllib.request
from collections import deque
from html import escape
from pathlib import Path

from wfc import MapCreator, UnfortunateSeedError, learn

HERE = Path(__file__).parent

# Sample tiles -> what is drawn. The sample keeps wall sides apart so WFC
# learns which side the floor is on; the render merges them back.
RENDER = {
    "━": "─", "┃": "│",
    "n": "+", "u": "+", "(": "+", ")": "+",
}
FLOOR, CORRIDOR, ROCK = ".", "#", " "
DOORS = {"n": (0, -1), "u": (0, 1), "(": (-1, 0), ")": (1, 0)}  # outward side
DOOR_TO_WALL = {"n": "─", "u": "━", "(": "│", ")": "┃"}

PALETTES = {
    "dark": {
        "bg": "#0d1117", "border": "#30363d", "text": "#e6edf3", "muted": "#8b949e",
        "wall": "#6e7681", "floor": "#30363d", "corridor": "#484f58", "door": "#d29922",
        "room": "#5fd39c", "hero": "#ffb454", "gold": "#e3b341", "stairs": "#79c0ff",
    },
    "light": {
        "bg": "#f6f8fa", "border": "#d0d7de", "text": "#1f2328", "muted": "#59636e",
        "wall": "#818b98", "floor": "#c8d1da", "corridor": "#9198a1", "door": "#9a6700",
        "room": "#1a7f4b", "hero": "#bc4c00", "gold": "#9a6700", "stairs": "#0969da",
    },
}


# ---------------------------------------------------------------- generation

def load_sample() -> list[str]:
    rows = (HERE / "sample.txt").read_text(encoding="utf-8").split("\n")
    rows = [r for r in rows if r.strip()]
    width = max(len(r) for r in rows) + 2
    # A ring of rock so only rock learns to touch the border.
    return [" " * width] + [(" " + r).ljust(width) for r in rows] + [" " * width]


def neighbors4(x, y, w, h):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield nx, ny


def clear_corridors(grid: list[list[str]]) -> None:
    """Drop WFC's corridors and doors; 1x1 tiles can't keep them connected.

    WFC is great at the room layout but corridors need global reachability,
    which the local adjacency rules can't express. ``connect`` digs them.
    """
    for row in grid:
        for x, c in enumerate(row):
            if c == CORRIDOR:
                row[x] = ROCK
            elif c in DOORS:
                row[x] = DOOR_TO_WALL[c]


WALL_TO_DOOR = {wall: door for door, wall in DOOR_TO_WALL.items()}


def room_exits(grid, room):
    """Non-corner wall cells of a room -> (wall x, y, outward x, y)."""
    h, w = len(grid), len(grid[0])
    exits = []
    for x in range(room["x0"], room["x1"] + 1):
        exits.append((x, room["y0"] - 1, x, room["y0"] - 2))
        exits.append((x, room["y1"] + 1, x, room["y1"] + 2))
    for y in range(room["y0"], room["y1"] + 1):
        exits.append((room["x0"] - 1, y, room["x0"] - 2, y))
        exits.append((room["x1"] + 1, y, room["x1"] + 2, y))
    return [e for e in exits if 0 <= e[2] < w and 0 <= e[3] < h and grid[e[3]][e[2]] in " #"]


def open_door(grid, x, y) -> None:
    grid[y][x] = WALL_TO_DOOR.get(grid[y][x], grid[y][x])


def hugs_wall(grid, x, y) -> bool:
    h, w = len(grid), len(grid[0])
    return any(grid[ny][nx] not in " #." for nx, ny in neighbors4(x, y, w, h))


def dig(grid, a, b) -> bool:
    """Dig the cheapest corridor between rooms a and b (turns cost extra)."""
    import heapq

    h, w = len(grid), len(grid[0])
    goals = {(ox, oy): (wx, wy) for wx, wy, ox, oy in room_exits(grid, b)}
    heap, best, prev = [], {}, {}
    for wx, wy, ox, oy in room_exits(grid, a):
        state = (ox, oy, -1)
        best[state] = 0
        prev[state] = ("start", wx, wy)
        heapq.heappush(heap, (0, ox, oy, -1))
    while heap:
        cost, x, y, d = heapq.heappop(heap)
        if best.get((x, y, d), 1e9) < cost:
            continue
        if (x, y) in goals:
            # Walk back, carve, and open both doors.
            state = (x, y, d)
            while True:
                sx, sy, _ = state
                grid[sy][sx] = CORRIDOR
                p = prev[state]
                if p[0] == "start":
                    open_door(grid, p[1], p[2])
                    break
                state = p
            open_door(grid, *goals[(x, y)])
            return True
        for nd, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
            nx, ny = x + dx, y + dy
            if not (1 <= nx < w - 1 and 1 <= ny < h - 1) or grid[ny][nx] not in " #":
                continue
            # Reuse corridors, keep turns rare, stay off other rooms' walls.
            step = 0.5 if grid[ny][nx] == CORRIDOR else 1
            step += 3 if d != -1 and nd != d else 0
            step += 2 if hugs_wall(grid, nx, ny) else 0
            state = (nx, ny, nd)
            if cost + step < best.get(state, 1e9):
                best[state] = cost + step
                prev[state] = (x, y, d)
                heapq.heappush(heap, (cost + step, nx, ny, nd))
    return False


def connect(grid, rooms) -> bool:
    """Join every room with a minimum spanning tree of corridors."""
    center = lambda r: ((r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2)
    linked, pending = [rooms[0]], rooms[1:]
    while pending:
        a, b = min(
            ((p, q) for p in linked for q in pending),
            key=lambda pq: abs(center(pq[0])[0] - center(pq[1])[0]) + abs(center(pq[0])[1] - center(pq[1])[1]),
        )
        if not dig(grid, a, b):
            return False
        linked.append(b)
        pending.remove(b)
    return True


def find_rooms(grid):
    h, w = len(grid), len(grid[0])
    seen, rooms = set(), []
    for y in range(h):
        for x in range(w):
            if grid[y][x] != FLOOR or (x, y) in seen:
                continue
            cells, queue = [], deque([(x, y)])
            seen.add((x, y))
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for n in neighbors4(cx, cy, w, h):
                    if n not in seen and grid[n[1]][n[0]] == FLOOR:
                        seen.add(n)
                        queue.append(n)
            xs, ys = [c[0] for c in cells], [c[1] for c in cells]
            rooms.append({"cells": cells, "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)})
    return rooms


def generate(seed_text: str, width: int, height: int, wanted: int, weights: dict | None = None):
    types = learn(load_sample())
    # Like CellType.weight in the Unity package: tune how often a tile is picked.
    for tile, weight in (weights or {}).items():
        types[tile].weight = weight
    base = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16)
    for attempt in range(200):
        try:
            rows = MapCreator(types, width, height, seed=base + attempt).create()
        except UnfortunateSeedError:
            continue
        grid = [list(r) for r in rows]
        clear_corridors(grid)
        rooms = find_rooms(grid)
        big = [r for r in rooms if len(r["cells"]) >= 6 and r["x1"] > r["x0"]]
        if len(big) < wanted + 1:
            continue
        rooms.sort(key=lambda r: -len(r["cells"]))
        if not connect(grid, rooms):
            continue
        big.sort(key=lambda r: -len(r["cells"]))
        return grid, big, attempt
    raise UnfortunateSeedError(f"no connected dungeon for seed {seed_text}")


# --------------------------------------------------------------------- data

def github_stats(user: str, repos: list[str], token: str | None) -> dict:
    aliases = "\n".join(
        f'r{i}: repository(owner: "{user}", name: "{name}") {{ pushedAt }}'
        for i, name in enumerate(repos)
    )
    query = f"""{{
      user(login: "{user}") {{
        createdAt
        contributionsCollection {{ contributionCalendar {{ totalContributions }} }}
        repositories(privacy: PUBLIC, ownerAffiliations: OWNER) {{ totalCount }}
        owned: repositories(first: 100, privacy: PUBLIC, ownerAffiliations: OWNER, isFork: false) {{
          nodes {{ stargazerCount }}
        }}
      }}
      {aliases}
    }}"""
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Authorization": f"bearer {token}", "User-Agent": "dungeon-readme"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)["data"]
    u = data["user"]
    return {
        "created": u["createdAt"][:10],
        "contributions": u["contributionsCollection"]["contributionCalendar"]["totalContributions"],
        "repos": u["repositories"]["totalCount"],
        "stars": sum(n["stargazerCount"] for n in u["owned"]["nodes"]),
        "pushed": {name: (data.get(f"r{i}") or {}).get("pushedAt") or "" for i, name in enumerate(repos)},
    }


OFFLINE_STATS = {
    "created": "2022-04-05", "contributions": 1714, "repos": 36, "stars": 2,
    "pushed": {"mudcrab": "2026-10-29"},
}


# ------------------------------------------------------------------- render

def place(grid, rooms, config, stats, rng):
    projects = config["rooms"]
    chosen = rooms[: len(projects) + 1]
    stairs_room = chosen.pop()
    chosen.sort(key=lambda r: (r["y0"], r["x0"]))  # reading order
    order = list(range(len(projects)))
    rng.shuffle(order)
    marks = {}
    project_room = {}
    for room, idx in zip(chosen, order):
        project_room[idx] = room
        marks[(room["x0"], room["y0"])] = ("room", projects[idx]["key"])

    with_repo = [i for i, p in enumerate(projects) if p.get("repo")]
    latest = max(with_repo, key=lambda i: stats["pushed"].get(projects[i]["repo"], ""))
    room = project_room[latest]
    cx, cy = (room["x0"] + room["x1"]) // 2, (room["y0"] + room["y1"]) // 2
    if (cx, cy) not in room["cells"] or (cx, cy) in marks:
        cx, cy = room["cells"][len(room["cells"]) // 2]
    marks[(cx, cy)] = ("hero", "@")

    sx, sy = stairs_room["x1"], stairs_room["y1"]
    marks[(sx, sy)] = ("stairs", ">")

    floors = [c for r in chosen + [stairs_room] for c in r["cells"] if c not in marks]
    for c in rng.sample(floors, min(stats["stars"], 9, len(floors))):
        marks[c] = ("gold", "$")
    return marks, projects[latest]


def classify(c: str) -> str:
    if c == FLOOR:
        return "floor"
    if c == CORRIDOR:
        return "corridor"
    if c in DOORS:
        return "door"
    if c == ROCK:
        return ""
    return "wall"


def render_svg(grid, marks, message, status, caption, palette) -> str:
    fs, cw, lh, pad = 14, 8.4, 16, 18
    # Crop empty rock rows so the box hugs the dungeon.
    used = [y for y, row in enumerate(grid) if "".join(row).strip()]
    first = used[0]
    grid = grid[first : used[-1] + 1]
    marks = {(x, y - first): v for (x, y), v in marks.items()}
    cols, rows = len(grid[0]), len(grid)
    top = pad + lh  # message line
    width = round(pad * 2 + cols * cw)
    height = round(top + 8 + rows * lh + 8 + lh * 2 + pad)
    p = palette
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(message)}">',
        f"<title>{escape(message)} {escape(status)}</title>",
        "<style>text{font-family:ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,"
        f"'Liberation Mono',monospace;font-size:{fs}px;white-space:pre}}"
        + "".join(f".{k}{{fill:{v}}}" for k, v in p.items())
        + ".b{font-weight:700}</style>",
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="8" '
        f'fill="{p["bg"]}" stroke="{p["border"]}"/>',
        f'<text x="{pad}" y="{top - 4}" class="text">{escape(message)}</text>',
    ]
    y0 = top + 8
    for y, row in enumerate(grid):
        runs: list[tuple[str, list[int], str]] = []
        for x, c in enumerate(row):
            if (x, y) in marks:
                cls, ch = marks[(x, y)]
                cls += " b"
            else:
                cls, ch = classify(c), RENDER.get(c, c)
            if not cls:
                continue
            if runs and runs[-1][0] == cls and runs[-1][1][-1] == x - 1:
                runs[-1][1].append(x)
                runs[-1] = (cls, runs[-1][1], runs[-1][2] + ch)
            else:
                runs.append((cls, [x], ch))
        if not runs:
            continue
        spans = "".join(
            f'<tspan class="{cls}" x="{" ".join(f"{pad + x * cw:.1f}" for x in xs)}">{escape(text)}</tspan>'
            for cls, xs, text in runs
        )
        out.append(f'<text y="{y0 + (y + 1) * lh - 4}">{spans}</text>')
    sy = y0 + rows * lh + 8 + lh
    out.append(f'<text x="{pad}" y="{sy}" class="muted">{status}</text>')
    out.append(f'<text x="{pad}" y="{sy + lh}" class="muted" font-size="12">{escape(caption)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist")
    ap.add_argument("--seed", default=dt.date.today().isoformat())
    ap.add_argument("--offline", action="store_true", help="skip the GitHub API, use sample stats")
    args = ap.parse_args()

    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    repos = [r["repo"] for r in config["rooms"] if r.get("repo")]
    stats = OFFLINE_STATS if args.offline else github_stats(config["user"], repos, os.environ.get("GITHUB_TOKEN"))

    grid, rooms, attempt = generate(
        args.seed, config["width"], config["height"], len(config["rooms"]), config.get("weights")
    )
    rng = random.Random(args.seed)
    marks, here = place(grid, rooms, config, stats, rng)

    today = dt.date.fromisoformat(args.seed)
    created = dt.date.fromisoformat(stats["created"])
    message = f"You see here: [{here['key']}] {here['name']}. The last thing {config['hero']} touched."
    b = lambda v: f'<tspan class="text b">{v}</tspan>'
    status = (
        f"{escape(config['hero'])} {escape(config['title'])}   "
        f"HP:{b(stats['contributions'])}  LVL:{b(stats['repos'])}  GOLD:{b(stats['stars'])}  "
        f"DEPTH:{b(today.year - created.year)}  TURN:{b((today - created).days)}"
    )
    caption = f"seed {args.seed} · generated with {config['user']}/WFC · @ you · > portfolio · $ stars"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, palette in PALETTES.items():
        (out / f"dungeon-{name}.svg").write_text(
            render_svg(grid, marks, message, status, caption, palette), encoding="utf-8"
        )
    print(f"seed {args.seed}: ok after {attempt + 1} attempt(s), {len(rooms)} linked rooms, @ in {here['name']}")


if __name__ == "__main__":
    main()
