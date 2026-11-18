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
import math
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


def shrink_room(grid, room, target: int) -> dict:
    """Shrink a rectangular room in place so its floor area is about ``target``."""
    w, h = room["x1"] - room["x0"] + 1, room["y1"] - room["y0"] + 1
    scale = math.sqrt(target / (w * h))
    nh = max(1, min(h, round(h * scale)))
    nw = max(1, min(w, round(target / nh)))
    if nw * nh < 2:  # leave space for the room number and @
        nw, nh = (2, 1) if w >= 2 else (1, 2)
    for y in range(room["y0"] - 1, room["y1"] + 2):
        for x in range(room["x0"] - 1, room["x1"] + 2):
            grid[y][x] = ROCK
    x0, y0 = room["x0"] + (w - nw) // 2, room["y0"] + (h - nh) // 2
    x1, y1 = x0 + nw - 1, y0 + nh - 1
    for x in range(x0, x1 + 1):
        grid[y0 - 1][x], grid[y1 + 1][x] = "─", "━"
    for y in range(y0, y1 + 1):
        grid[y][x0 - 1], grid[y][x1 + 1] = "│", "┃"
        for x in range(x0, x1 + 1):
            grid[y][x] = FLOOR
    grid[y0 - 1][x0 - 1], grid[y0 - 1][x1 + 1] = "┌", "┐"
    grid[y1 + 1][x0 - 1], grid[y1 + 1][x1 + 1] = "└", "┘"
    cells = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    return {"cells": cells, "x0": x0, "y0": y0, "x1": x1, "y1": y1}


def generate(seed_text: str, width: int, height: int, commits: list[int], weights: dict | None = None):
    """WFC lays out the rooms; the busiest repo gets the biggest one.

    Rooms are then shrunk so their floor area follows each repo's share of
    commits, and finally joined with corridors.
    """
    types = learn(load_sample())
    # Like CellType.weight in the Unity package: tune how often a tile is picked.
    for tile, weight in (weights or {}).items():
        types[tile].weight = weight
    base = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16)
    needed = max(len(commits), 1) + 1  # + the room with the stairs
    for attempt in range(200):
        try:
            rows = MapCreator(types, width, height, seed=base + attempt).create()
        except UnfortunateSeedError:
            continue
        grid = [list(r) for r in rows]
        clear_corridors(grid)
        big = sorted(
            (r for r in find_rooms(grid) if len(r["cells"]) >= 6 and r["x1"] > r["x0"]),
            key=lambda r: -len(r["cells"]),
        )
        if len(big) < needed:
            continue
        stairs = big[needed - 1]
        assigned = big[: needed - 1]
        if commits:
            largest = len(assigned[0]["cells"])
            for i, room in enumerate(assigned):
                target = max(2, round(largest * commits[i] / commits[0]))
                if target < len(room["cells"]):
                    assigned[i] = shrink_room(grid, room, target)
        if not connect(grid, sorted(find_rooms(grid), key=lambda r: -len(r["cells"]))):
            continue
        return grid, assigned, stairs, attempt
    raise UnfortunateSeedError(f"no connected dungeon for seed {seed_text}")


# --------------------------------------------------------------------- data

def graphql(query: str, token: str | None) -> dict:
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Authorization": f"bearer {token}", "User-Agent": "dungeon-readme"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    if body.get("errors"):
        raise RuntimeError(body["errors"])
    return body["data"]


def github_stats(user: str, today: dt.date, lookback: int, token: str | None) -> dict:
    since = today - dt.timedelta(days=lookback)
    u = graphql(
        f"""{{
      user(login: "{user}") {{
        createdAt
        contributionsCollection {{ contributionCalendar {{ totalContributions }} }}
        repositories(privacy: PUBLIC, ownerAffiliations: OWNER) {{ totalCount }}
        owned: repositories(first: 100, privacy: PUBLIC, ownerAffiliations: OWNER, isFork: false) {{
          nodes {{ stargazerCount }}
        }}
        recent: contributionsCollection(from: "{since}T00:00:00Z", to: "{today}T00:00:00Z") {{
          commitContributionsByRepository(maxRepositories: 100) {{
            repository {{ name nameWithOwner url isPrivate owner {{ login }} }}
            contributions(first: 100) {{ nodes {{ occurredAt commitCount }} }}
          }}
        }}
      }}
    }}""",
        token,
    )["user"]
    # date -> {repo: (label, url, commits)}
    days: dict[str, dict[str, list]] = {}
    for entry in u["recent"]["commitContributionsByRepository"]:
        repo = entry["repository"]
        if repo["isPrivate"]:
            continue
        label = repo["name"] if repo["owner"]["login"].lower() == user.lower() else repo["nameWithOwner"]
        for node in entry["contributions"]["nodes"]:
            day = days.setdefault(node["occurredAt"][:10], {})
            day.setdefault(repo["nameWithOwner"], [label, repo["url"], 0])[2] += node["commitCount"]
    last_day = max(days, default=None)
    activity = sorted(days.get(last_day, {}).values(), key=lambda r: (-r[2], r[0].lower()))
    return {
        "created": u["createdAt"][:10],
        "contributions": u["contributionsCollection"]["contributionCalendar"]["totalContributions"],
        "repos": u["repositories"]["totalCount"],
        "stars": sum(n["stargazerCount"] for n in u["owned"]["nodes"]),
        "day": last_day,
        "activity": [{"name": n, "url": url, "commits": c} for n, url, c in activity],
    }


OFFLINE_STATS = {
    "created": "2022-04-05", "contributions": 1714, "repos": 29, "stars": 2, "day": None,
    "activity": [
        {"name": "mudcrab", "url": "https://github.com/Ggargur/mudcrab", "commits": 12},
        {"name": "WFC", "url": "https://github.com/Ggargur/WFC", "commits": 5},
        {"name": "Ggargur", "url": "https://github.com/Ggargur/Ggargur", "commits": 2},
        {"name": "double-u", "url": "https://github.com/Ggargur/double-u", "commits": 1},
    ],
}


# ------------------------------------------------------------------- place

def place(assigned, stairs, stars: int, rng):
    marks = {}
    for i, room in enumerate(assigned):
        marks[(room["x0"], room["y0"])] = ("room", str(i + 1))
    hero_room = assigned[0] if assigned else stairs
    cx, cy = (hero_room["x0"] + hero_room["x1"]) // 2, (hero_room["y0"] + hero_room["y1"]) // 2
    if (cx, cy) in marks:
        cx, cy = hero_room["x1"], hero_room["y1"]
    marks[(cx, cy)] = ("hero", "@")
    sx, sy = stairs["x1"], stairs["y1"]
    if (sx, sy) in marks:
        sx = stairs["x0"]
    marks[(sx, sy)] = ("stairs", ">")
    floors = [c for r in assigned + [stairs] for c in r["cells"] if c not in marks]
    for c in rng.sample(floors, min(stars, 9, len(floors))):
        marks[c] = ("gold", "$")
    return marks


# ------------------------------------------------------------------- render

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


def legend(rooms: list[dict], extra: list[dict], portfolio: str) -> str:
    parts = [f"`{i + 1}` [{r['name']}]({r['url']}) ×{r['commits']}" for i, r in enumerate(rooms)]
    if all(r["url"] != portfolio for r in rooms):
        parts.append(f"`>` [portfolio]({portfolio})")
    text = " &nbsp;·&nbsp; ".join(parts)
    if extra:
        text += f"<br>\n<sub>+{len(extra)} more: " + ", ".join(f"[{r['name']}]({r['url']})" for r in extra) + "</sub>"
    return text


def update_readme(path: Path, text: str) -> bool:
    start, end = "<!-- dungeon:start -->", "<!-- dungeon:end -->"
    readme = path.read_text(encoding="utf-8")
    i, j = readme.index(start) + len(start), readme.index(end)
    new = readme[:i] + "\n" + text + "\n" + readme[j:]
    if new == readme:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist")
    ap.add_argument("--seed", default=dt.date.today().isoformat())
    ap.add_argument("--readme", type=Path, help="rewrite the legend between the dungeon markers")
    ap.add_argument("--offline", action="store_true", help="skip the GitHub API, use sample stats")
    args = ap.parse_args()

    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    today = dt.date.fromisoformat(args.seed)
    if args.offline:
        stats = dict(OFFLINE_STATS, day=(today - dt.timedelta(days=1)).isoformat())
    else:
        stats = github_stats(config["user"], today, config["lookback_days"], os.environ.get("GITHUB_TOKEN"))

    rooms = stats["activity"][: config["max_rooms"]]
    extra = stats["activity"][config["max_rooms"] :]
    grid, assigned, stairs, attempt = generate(
        args.seed, config["width"], config["height"], [r["commits"] for r in rooms], config.get("weights")
    )
    marks = place(assigned, stairs, stats["stars"], random.Random(args.seed))

    hero = config["hero"]
    if not rooms:
        message = f"{hero} has not explored anything lately."
    else:
        top = rooms[0]
        commits = f"{top['commits']} commit{'s' if top['commits'] != 1 else ''}"
        if stats["day"] == (today - dt.timedelta(days=1)).isoformat():
            message = f"You see here: [1] {top['name']}. {commits} yesterday."
        else:
            message = f"Last seen on {stats['day']} in [1] {top['name']}. {commits}."

    created = dt.date.fromisoformat(stats["created"])
    b = lambda v: f'<tspan class="text b">{v}</tspan>'
    status = (
        f"{escape(hero)} {escape(config['title'])}   "
        f"HP:{b(stats['contributions'])}  LVL:{b(stats['repos'])}  GOLD:{b(stats['stars'])}  "
        f"DEPTH:{b(today.year - created.year)}  TURN:{b((today - created).days)}"
    )
    caption = f"seed {args.seed} · {config['user']}/WFC · room size = commits"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, palette in PALETTES.items():
        (out / f"dungeon-{name}.svg").write_text(
            render_svg(grid, marks, message, status, caption, palette), encoding="utf-8"
        )
    changed = args.readme and update_readme(args.readme, legend(rooms, extra, config["portfolio"]))
    names = ", ".join(f"{r['name']}×{r['commits']}" for r in rooms) or "none"
    print(f"seed {args.seed}: attempt {attempt + 1}, day {stats['day']}, rooms: {names}, readme changed: {bool(changed)}")


if __name__ == "__main__":
    main()
