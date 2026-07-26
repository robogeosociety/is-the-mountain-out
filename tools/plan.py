import sys
import random
import argparse
from datetime import datetime, timedelta, UTC
from typing import List

try:
    from astral import LocationInfo
    from astral.sun import sun
except ImportError:
    print("Error: 'astral' library not found. Run with 'uv pip install astral'")
    sys.exit(1)


class CapturePlan:
    def __init__(self, lat: float, lon: float, name: str = "Mount Rainier"):
        self.location = LocationInfo(name, "USA", "UTC", lat, lon)

    def generate(
        self, start_time: datetime, days: int = 30, jitter: int = 300
    ) -> List[str]:
        intervals = []
        current = start_time

        for d_offset in range(days + 1):
            day = start_time.date() + timedelta(days=d_offset)
            s = sun(self.location.observer, date=day)

            # Key segments for each day
            # 1. Early Night (if day starts before sunrise)
            # 2. Dawn Golden Hour (Sunrise +/- 45m)
            # 3. Standard Day (Dawn end to Dusk start)
            # 4. Dusk Golden Hour (Sunset +/- 45m)
            # 5. Night (Sunset + 45m to next Dawn start)

            segments = [
                {
                    "name": "GOLDEN",
                    "start": s["sunrise"] - timedelta(minutes=45),
                    "end": s["sunrise"] + timedelta(minutes=45),
                    "gap": 600,
                },
                {
                    "name": "DAY",
                    "start": s["sunrise"] + timedelta(minutes=46),
                    "end": s["sunset"] - timedelta(minutes=46),
                    "gap": 1800,
                },
                {
                    "name": "GOLDEN",
                    "start": s["sunset"] - timedelta(minutes=45),
                    "end": s["sunset"] + timedelta(minutes=45),
                    "gap": 600,
                },
                {
                    "name": "NIGHT",
                    "start": s["sunset"] + timedelta(minutes=46),
                    "end": s["sunset"] + timedelta(hours=8),
                    "gap": 28800,
                },  # One anchor at night
            ]

            for seg in sorted(segments, key=lambda x: x["start"]):
                if seg["end"] < current:
                    continue

                # Move 'current' to start of segment if we are before it
                if current < seg["start"]:
                    wait = int((seg["start"] - current).total_seconds())
                    if wait > 0:
                        intervals.append(f"{wait}s")
                        current = seg["start"]

                # Fill segment
                while current < seg["end"]:
                    wait = seg["gap"] + random.randint(-jitter, jitter)
                    # Don't overshoot segment too much
                    if current + timedelta(seconds=wait) > seg["end"] + timedelta(
                        minutes=5
                    ):
                        break
                    intervals.append(f"{wait}s")
                    current += timedelta(seconds=wait)

            if current > start_time + timedelta(days=days):
                break

        intervals.append("stop")
        return intervals

    def simulate(self, start_time: datetime, intervals: List[str]):
        print(
            f"\n📈 PLAN SIMULATION (Start: {start_time.strftime('%Y-%m-%d %H:%M')} UTC)"
        )
        print("-" * 80)
        current = start_time

        for i, step in enumerate(intervals):
            if step == "stop":
                break
            wait_sec = int(step[:-1])
            if i % 100 == 0 or i < 10:
                print(
                    f"Capture {i + 1:<4} | {current.strftime('%Y-%m-%d %H:%M'):<20} | Next: {step}"
                )
            current += timedelta(seconds=wait_sec)

        print("-" * 80)
        print(
            f"Total captures: {len(intervals) - 1} over {(current - start_time).days} days"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    planner = CapturePlan(47.6533, -122.3091)
    # Start at a clean hour
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    steps = planner.generate(now, days=args.days)

    if args.simulate:
        planner.simulate(now, steps)
    else:
        print(" ".join([f"--plan-steps {s}" for s in steps]))
