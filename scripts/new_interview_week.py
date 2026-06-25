"""Create a new interview-prep week scaffold.

Usage:
    py scripts/new_interview_week.py 2 --start-date 2026-06-29

The script creates:
    docs/interview-prep/weeks/week-XX.md
    docs/interview-prep/weeks/week-XX/<weekday-topic>.md
    docs/interview-prep/quant-7-in-30/week-XX/<day>.md

It also adds the week to docs/interview-prep/weeks/index.md.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INTERVIEW_PREP = ROOT / "docs" / "interview-prep"
WEEKS_DIR = INTERVIEW_PREP / "weeks"
QUANT_DIR = INTERVIEW_PREP / "quant-7-in-30"

WEEKDAYS = [
    {
        "day": "Monday",
        "slug": "monday",
        "topic_slug": "monday-statistics",
        "theme": "Statistics",
        "focus": "statistics, probability, and experimentation",
        "title_suffix": "Statistics",
    },
    {
        "day": "Tuesday",
        "slug": "tuesday",
        "topic_slug": "tuesday-sql",
        "theme": "SQL",
        "focus": "SQL, analytics, and data manipulation",
        "title_suffix": "SQL",
    },
    {
        "day": "Wednesday",
        "slug": "wednesday",
        "topic_slug": "wednesday-model-implementation",
        "theme": "Model Implementation",
        "focus": "models and algorithms from scratch",
        "title_suffix": "Model Implementation",
    },
    {
        "day": "Thursday",
        "slug": "thursday",
        "topic_slug": "thursday-ml-system-design",
        "theme": "ML System Design",
        "focus": "ML system design and production architecture",
        "title_suffix": "ML System Design",
    },
    {
        "day": "Friday",
        "slug": "friday",
        "topic_slug": "friday-applied-case-study",
        "theme": "Applied Case Study",
        "focus": "applied data science, product reasoning, and weekly review",
        "title_suffix": "Applied Case Study",
    },
]

ALL_DAYS = [
    {"day": "Monday", "slug": "monday"},
    {"day": "Tuesday", "slug": "tuesday"},
    {"day": "Wednesday", "slug": "wednesday"},
    {"day": "Thursday", "slug": "thursday"},
    {"day": "Friday", "slug": "friday"},
    {"day": "Saturday", "slug": "saturday"},
    {"day": "Sunday", "slug": "sunday"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an interview-prep week.")
    parser.add_argument("week", type=int, help="Week number, for example 2.")
    parser.add_argument(
        "--start-date",
        help="Monday date for the week in YYYY-MM-DD format. Defaults to placeholders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing week files.",
    )
    return parser.parse_args()


def parse_start_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def week_label(week: int) -> str:
    return f"Week {week:02d}"


def week_slug(week: int) -> str:
    return f"week-{week:02d}"


def day_dates(start: date | None) -> dict[str, str]:
    if start is None:
        return {item["slug"]: "YYYY-MM-DD" for item in ALL_DAYS}
    return {
        item["slug"]: (start + timedelta(days=index)).isoformat()
        for index, item in enumerate(ALL_DAYS)
    }


def write_file(path: Path, content: str, force: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        print(f"Skipped existing file: {path.relative_to(ROOT)}")
        return False
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote: {path.relative_to(ROOT)}")
    return True


def update_weeks_index(week: int, force: bool) -> None:
    path = WEEKS_DIR / "index.md"
    entry = f"- [{week_label(week)}]({week_slug(week)}.md)"

    if not path.exists():
        content = f"""---
title: Interview Prep Weeks
description: Weekly interview-preparation overview pages.
---

# Interview Prep Weeks

Each week links to its topic posts and 7-in-30 problem pages.

<!-- weeks-list:start -->
{entry}
<!-- weeks-list:end -->
"""
        write_file(path, content, force=True)
        return

    content = path.read_text(encoding="utf-8")
    if entry in content and not force:
        print(f"Weeks index already contains: {entry}")
        return

    start_marker = "<!-- weeks-list:start -->"
    end_marker = "<!-- weeks-list:end -->"

    if start_marker in content and end_marker in content:
        before, rest = content.split(start_marker, 1)
        current, after = rest.split(end_marker, 1)
        lines = [line.strip() for line in current.splitlines() if line.strip()]
        if entry not in lines:
            lines.append(entry)
        new_block = "\n".join(sorted(lines)) if lines else entry
        updated = f"{before}{start_marker}\n{new_block}\n{end_marker}{after}"
    else:
        updated = content.rstrip() + "\n\n" + entry + "\n"

    path.write_text(updated, encoding="utf-8")
    print(f"Updated: {path.relative_to(ROOT)}")


def week_overview_template(week: int, dates: dict[str, str]) -> str:
    label = week_label(week)
    slug = week_slug(week)

    topic_rows = []
    for item in WEEKDAYS:
        topic_rows.append(
            "| {day} | {theme} | TBD | [Post]({slug}/{topic_slug}.md) | "
            "[7-in-30](../quant-7-in-30/{slug}/{day_slug}.md) | 0 / 7, 30 min |".format(
                day=item["day"],
                theme=item["theme"],
                slug=slug,
                topic_slug=item["topic_slug"],
                day_slug=item["slug"],
            )
        )

    quant_rows = []
    for item in ALL_DAYS:
        quant_rows.append(
            "| {day} | {date} | 0 / 7 | 30 min | "
            "[Problems](../quant-7-in-30/{slug}/{day_slug}.md) | TBD |".format(
                day=item["day"],
                date=dates[item["slug"]],
                slug=slug,
                day_slug=item["slug"],
            )
        )

    return f"""---
title: Interview Prep {label}
description: {label} overview for interview preparation.
hide:
  - navigation
---

# Interview Prep: {label}

[Back to weeks](index.md)

This overview is the hub for {label}. It tracks topic posts, daily 7-in-30 pages, and performance.

## Daily Topic Posts

| Day | Focus | Topic | Topic post | 7-in-30 | Performance |
|---|---|---|---|---|---|
{chr(10).join(topic_rows)}

## Weekly 7-in-30

| Day | Date | Score | Time | Problem page | Main miss |
|---|---|---:|---:|---|---|
{chr(10).join(quant_rows)}

## Weekly Scorecard

| Area | Target | Completed | Notes |
|---|---:|---:|---|
| Quant problems | 49 | 0 |  |
| Quant blocks | 7 | 0 |  |
| Statistics sessions | 1 | 0 |  |
| SQL sessions | 1 | 0 |  |
| Model implementations | 1 | 0 |  |
| System designs | 1 | 0 |  |
| Applied cases | 1 | 0 |  |

## Weekly Reflection

### What improved?

TBD

### What repeatedly caused errors?

TBD

### What changes next week?

TBD
"""


def topic_template(week: int, item: dict[str, str]) -> str:
    label = week_label(week)
    slug = week_slug(week)
    title = f"{label} {item['day']} - {item['title_suffix']}"
    focus_sentence = item["focus"][:1].upper() + item["focus"][1:]

    return f"""---
title: {title}
description: {label} {item['day']} {item['focus']} practice.
hide:
  - navigation
---

# {label} {item['day']}: {item['title_suffix']}

**Week:** [{label}](../{slug}.md)  
**7-in-30:** [{item['day']} quant problems](../../quant-7-in-30/{slug}/{item['slug']}.md)  
**Quant performance:** 0 / 7 in 30 minutes  
**Topic:** TBD

## Focus

{focus_sentence}.

## Work

TBD

## Result

TBD

## Takeaway

TBD
"""


def quant_template(week: int, item: dict[str, str]) -> str:
    label = week_label(week)
    slug = week_slug(week)
    topic = next((day for day in WEEKDAYS if day["slug"] == item["slug"]), None)
    topic_line = ""
    if topic:
        topic_line = (
            f"**Topic post:** [{topic['title_suffix']}](../../weeks/{slug}/{topic['topic_slug']}.md)  \n"
        )

    problems = []
    for number in range(1, 8):
        problems.append(
            f"""## Problem {number}

### Problem

Problem statement.

### My solution

Solution attempt.
"""
        )

    return f"""---
title: Quant 7-in-30 {label} {item['day']}
description: {item['day']} problem set for {label} of quant 7-in-30 interview practice.
hide:
  - navigation
---

# Quant 7-in-30: {label} {item['day']}

**Timer:** 30 minutes  
**Score:** 0 / 7  
**Week:** [{label}](../../weeks/{slug}.md)  
{topic_line.rstrip()}

{chr(10).join(problems)}
"""


def main() -> None:
    args = parse_args()
    if args.week < 1:
        raise SystemExit("Week must be a positive integer.")

    start = parse_start_date(args.start_date)
    dates = day_dates(start)
    label = week_label(args.week)
    slug = week_slug(args.week)

    write_file(
        WEEKS_DIR / f"{slug}.md",
        week_overview_template(args.week, dates),
        force=args.force,
    )

    for item in WEEKDAYS:
        write_file(
            WEEKS_DIR / slug / f"{item['topic_slug']}.md",
            topic_template(args.week, item),
            force=args.force,
        )

    for item in ALL_DAYS:
        write_file(
            QUANT_DIR / slug / f"{item['slug']}.md",
            quant_template(args.week, item),
            force=args.force,
        )

    update_weeks_index(args.week, force=args.force)

    print()
    print(f"Created scaffold for {label}.")
    print(f"Open docs/interview-prep/weeks/{slug}.md to fill in scores and links.")


if __name__ == "__main__":
    main()
