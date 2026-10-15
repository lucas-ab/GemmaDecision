"""Snake as a sequence of typed decisions. Standard library only (PIL optional, for image observations), so
other models' environments (e.g. Julia-1's) can import it without this project's dependencies.

Each tick is one `choice` question over the legal moves (the reverse of the current heading is excluded).
Observation modes for the decision row:
  grid   ASCII board + coordinates; the model must work out the geometry itself
  hints  same board, and every option states its immediate consequence (wall / body / food distance)
  image  the board as a picture (<|image|>), plus the same one-line text summary
"""

from __future__ import annotations

import io
import random
from collections import deque
from dataclasses import dataclass, field

MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
REVERSE = {"up": "down", "down": "up", "left": "right", "right": "left"}


@dataclass
class SnakeGame:
    size: int = 10
    seed: int = 0
    starve_limit: int = 100  # steps without food before the game ends (stops endless loops)
    snake: list[tuple[int, int]] = field(default_factory=list)  # head first
    heading: str = "right"
    food: tuple[int, int] | None = None
    score: int = 0
    steps: int = 0
    since_food: int = 0
    alive: bool = True
    death: str = ""

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        mid = self.size // 2
        self.snake = [(mid, 2), (mid, 1), (mid, 0)]
        self._place_food()

    def _place_food(self):
        free = [(r, c) for r in range(self.size) for c in range(self.size) if (r, c) not in self.snake]
        self.food = self.rng.choice(free) if free else None

    def legal_moves(self) -> list[str]:
        return [m for m in MOVES if m != REVERSE[self.heading]]

    def outcome(self, move: str) -> str:
        """'wall', 'body', 'food' or 'free' for the cell the head would enter."""
        r, c = self.next_cell(move)
        if not (0 <= r < self.size and 0 <= c < self.size):
            return "wall"
        eating = (r, c) == self.food
        body = self.snake if eating else self.snake[:-1]  # the tail moves away unless the snake grows
        if (r, c) in body:
            return "body"
        return "food" if eating else "free"

    def next_cell(self, move: str) -> tuple[int, int]:
        dr, dc = MOVES[move]
        return self.snake[0][0] + dr, self.snake[0][1] + dc

    def step(self, move: str):
        if move not in self.legal_moves():
            move = self.heading  # an illegal reverse keeps going straight
        what = self.outcome(move)
        self.steps += 1
        self.heading = move
        if what in ("wall", "body"):
            self.alive, self.death = False, what
            return
        cell = self.next_cell(move)
        self.snake.insert(0, cell)
        if what == "food":
            self.score += 1
            self.since_food = 0
            self._place_food()
            if self.food is None:
                self.alive, self.death = False, "won"
        else:
            self.snake.pop()
            self.since_food += 1
            if self.since_food >= self.starve_limit:
                self.alive, self.death = False, "starved"

    # ------------------------------------------------------------------ observations

    def grid_text(self) -> str:
        rows = ["   " + " ".join(str(c % 10) for c in range(self.size))]
        body = set(self.snake[1:])
        for r in range(self.size):
            cells = []
            for c in range(self.size):
                cells.append("H" if (r, c) == self.snake[0] else "o" if (r, c) in body
                             else "*" if (r, c) == self.food else ".")
            rows.append(f"{r % 10:>2} " + " ".join(cells))
        return "\n".join(rows)

    def summary(self) -> str:
        hr, hc = self.snake[0]
        fr, fc = self.food
        return (f"Snake game on a {self.size}x{self.size} board (row 0 is the top, column 0 is the left). "
                f"The head is at row {hr}, column {hc}, heading {self.heading}; the snake is {len(self.snake)} cells long. "
                f"The food is at row {fr}, column {fc}. Moving into a wall or the snake's own body ends the game.")

    def image_bytes(self, cell: int = 24) -> bytes:
        from PIL import Image, ImageDraw

        n = self.size
        img = Image.new("RGB", (n * cell, n * cell), (24, 24, 24))
        d = ImageDraw.Draw(img)
        for r in range(n):
            for c in range(n):
                d.rectangle([c * cell, r * cell, (c + 1) * cell - 1, (r + 1) * cell - 1], outline=(48, 48, 48))
        for i, (r, c) in enumerate(self.snake):
            color = (60, 200, 90) if i == 0 else (30, 120, 60)
            d.rectangle([c * cell + 2, r * cell + 2, (c + 1) * cell - 3, (r + 1) * cell - 3], fill=color)
        fr, fc = self.food
        d.ellipse([fc * cell + 4, fr * cell + 4, (fc + 1) * cell - 5, (fr + 1) * cell - 5], fill=(220, 60, 60))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    def describe(self, move: str, hints: bool) -> str:
        r, c = self.next_cell(move)
        base = {"up": "Move up", "down": "Move down", "left": "Move left", "right": "Move right"}[move]
        base += f" to row {r}, column {c}"
        if not hints:
            return base + "."
        what = self.outcome(move)
        if what in ("wall", "body"):
            return f"{base}: hits the {'wall' if what == 'wall' else 'snake body'} and ends the game."
        if what == "food":
            return f"{base}: eats the food."
        hr, hc = self.snake[0]
        before = abs(hr - self.food[0]) + abs(hc - self.food[1])
        after = abs(r - self.food[0]) + abs(c - self.food[1])
        space = reachable_after(self, move)
        return f"{base}: safe; food distance {before} -> {after}; {space} free cells reachable afterwards."

    def decision(self, mode: str = "grid") -> dict:
        """One typed-decisions row: {state, questions, images}."""
        moves = self.legal_moves()
        q = {"move": {"type": "choice",
                      "instructions": "Which way should the snake move next to stay alive and reach the food?",
                      "criteria": {m: self.describe(m, hints=(mode == "hints")) for m in moves}}}
        if mode == "image":
            return {"state": f"{self.summary()}\nBoard: <|image|>", "questions": q, "images": [self.image_bytes()]}
        return {"state": f"{self.summary()}\nBoard (H = head, o = body, * = food):\n{self.grid_text()}",
                "questions": q, "images": []}


# ---------------------------------------------------------------------- reference policies


def _free_cells(game: SnakeGame, snake: list[tuple[int, int]]):
    blocked = set(snake[:-1])
    return lambda cell: 0 <= cell[0] < game.size and 0 <= cell[1] < game.size and cell not in blocked


def reachable_after(game: SnakeGame, move: str) -> int:
    """Cells reachable by flood fill from the head after making `move` (0 if the move is fatal)."""
    if game.outcome(move) in ("wall", "body"):
        return 0
    head = game.next_cell(move)
    snake = [head] + game.snake[: len(game.snake) - (0 if head == game.food else 1)]
    ok = _free_cells(game, snake)
    seen, todo = {head}, deque([head])
    while todo:
        r, c = todo.popleft()
        for dr, dc in MOVES.values():
            nxt = (r + dr, c + dc)
            if nxt not in seen and ok(nxt):
                seen.add(nxt)
                todo.append(nxt)
    return len(seen) - 1


def _bfs_first_move(game: SnakeGame) -> str | None:
    """First move of a shortest path to the food that avoids the body (tail treated as moving)."""
    ok = _free_cells(game, game.snake)
    start = game.snake[0]
    prev = {start: None}
    todo = deque([start])
    while todo:
        cur = todo.popleft()
        if cur == game.food:
            while prev[cur] != start:
                cur = prev[cur]
            for m, (dr, dc) in MOVES.items():
                if (start[0] + dr, start[1] + dc) == cur:
                    return m
        for dr, dc in MOVES.values():
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt not in prev and ok(nxt):
                prev[nxt] = cur
                todo.append(nxt)
    return None


def teacher(game: SnakeGame) -> str:
    """BFS to the food when that move keeps enough room; otherwise the move with the most reachable space."""
    legal = game.legal_moves()
    first = _bfs_first_move(game)
    if first in legal and reachable_after(game, first) >= len(game.snake):
        return first
    return max(legal, key=lambda m: (reachable_after(game, m), game.outcome(m) == "food"))


def greedy(game: SnakeGame) -> str:
    """Closest to the food among moves that are not immediately fatal."""
    legal = game.legal_moves()
    safe = [m for m in legal if game.outcome(m) not in ("wall", "body")] or legal
    def dist(m):
        r, c = game.next_cell(m)
        return abs(r - game.food[0]) + abs(c - game.food[1])
    return min(safe, key=dist)


def make_random(seed: int = 0):
    """Uniform over legal moves, with its own RNG (the game's RNG places food and must stay identical)."""
    rng = random.Random(f"random-policy-{seed}")
    return lambda game: rng.choice(game.legal_moves())


# ---------------------------------------------------------------------- evaluation loop


def play(policy, seed: int, size: int = 10, max_steps: int = 500, record: bool = False) -> dict:
    """policy(game) -> move. Returns score/steps/death, agreement with the teacher, and optionally frames."""
    game = SnakeGame(size=size, seed=seed)
    agree, frames = 0, []
    while game.alive and game.steps < max_steps:
        want = teacher(game)
        move = policy(game)
        agree += move == want
        if record:
            frames.append({"snake": list(game.snake), "food": game.food, "move": move, "teacher": want})
        game.step(move)
    out = {"seed": seed, "score": game.score, "steps": game.steps, "death": game.death or "max_steps",
           "teacher_agreement": agree / max(1, game.steps)}
    if record:
        out["frames"] = frames
    return out
