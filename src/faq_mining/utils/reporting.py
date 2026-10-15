# Run reporting every stage shares: a stats summary and periodic progress lines

import time
from collections import Counter


class Stats(Counter):
    # Counter that renders as an aligned summary block
    # Because it's a Counter, a missing key counts as 0,
    # so a new drop reason can't crash a long run with a KeyError

    def render(self, indent: str = "  ") -> str:
        if not self:
            return f"{indent}(nothing recorded)"
        width = max(len(k) for k in self)
        return "\n".join(f"{indent}{k:<{width}}  {v:,}" for k, v in self.items())


class Progress:
    # Gives periodic updates on progress
    # In Stage 0, position is in bytes while `every` counts messages,
    # so the two use different units

    def __init__(
        self, total: float, every: int = 2000, start: float = 0.0, unit: str = "MB", scale: float = 1e6
    ) -> None:
        self.total = max(total, 1)
        self.every = every
        self.start = start
        self.unit = unit
        self.scale = scale
        self.ticks = 0
        self.t0 = time.time()

    def tick(self, position: float, note: str = "") -> str | None:
        # count one item, emit a line every `every` items
        self.ticks += 1
        if self.ticks % self.every:
            return None
        elapsed = time.time() - self.t0
        rate = (position - self.start) / elapsed / self.scale if elapsed else 0.0
        remaining = (self.total - position) / (rate * self.scale) if rate else 0.0
        line = (
            f"{self.ticks:>7,} | {100.0 * position / self.total:5.1f}% | "
            f"{rate:6.1f} {self.unit}/s | ETA {remaining / 60:5.1f} min" + (f" | {note}" if note else "")
        )
        print(line, flush=True)
        return line

    def elapsed_min(self) -> float:
        return (time.time() - self.t0) / 60
