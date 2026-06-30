"""Date-completeness checks and calendar/period helpers."""

from __future__ import annotations

import csv
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import holidays


def _add_month(current: date) -> date:
    year = current.year + (1 if current.month == 12 else 0)
    month = 1 if current.month == 12 else current.month + 1
    return date(year, month, 1)


def _period_key(value: date, granularity: str) -> str:
    if granularity == "day":
        return value.isoformat()
    if granularity == "week":
        iso = value.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if granularity == "month":
        return f"{value.year:04d}-{value.month:02d}"
    return value.isoformat()


def _extract_periods(text: str, granularity: str) -> set[str]:
    periods = set()
    if granularity == "week":
        periods.update(match.upper() for match in re.findall(r"\d{4}-W\d{2}", text, flags=re.IGNORECASE))
    if granularity == "month":
        periods.update(re.findall(r"\d{4}-\d{2}", text))
    for match in re.findall(r"\d{4}-\d{2}-\d{2}", text):
        parsed = date.fromisoformat(match)
        periods.add(_period_key(parsed, granularity))
    return periods


def _calendar_holidays(calendar: str, start: date, end: date) -> set[str]:
    try:
        import holidays as holidays_lib
    except ImportError:
        return set()

    calendar_map = {
        "us_business_day": "US",
        "ru_business_day": "RU",
    }
    country = calendar_map.get(calendar)
    if not country:
        return set()
    years = range(start.year, end.year + 1)
    try:
        return {day.isoformat() for day in holidays_lib.country_holidays(country, years=years)}
    except Exception:
        return set()


def check_date_completeness(
    start_date: str,
    end_date: str,
    actual_items: list[str],
    granularity: str = "day",
    calendar: str = "calendar",
    holidays: list[str] | None = None,
) -> dict:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        return {"error": "end_date is before start_date"}
    if granularity not in {"day", "week", "month"}:
        return {"error": "Supported granularities: day, week, month"}
    supported_calendars = {"calendar", "business_day", "crypto_24_7", "forex_weekday", "us_business_day", "ru_business_day"}
    if calendar not in supported_calendars:
        return {"error": f"Supported calendars: {', '.join(sorted(supported_calendars))}"}
    holiday_set = set(holidays or []) | _calendar_holidays(calendar, start, end)

    expected = []
    if granularity == "month":
        current = date(start.year, start.month, 1)
        end_month = date(end.year, end.month, 1)
        while current <= end_month:
            expected.append(_period_key(current, granularity))
            current = _add_month(current)
    else:
        current = start
        seen = set()
        step = timedelta(days=1 if granularity == "day" else 7)
        while current <= end:
            key = _period_key(current, granularity)
            include = True
            if granularity == "day" and calendar in {"business_day", "forex_weekday", "us_business_day", "ru_business_day"}:
                include = current.weekday() < 5 and current.isoformat() not in holiday_set
            if include and key not in seen:
                expected.append(key)
                seen.add(key)
            current += step

    actual = set()
    for item in actual_items:
        text = str(item)
        actual.update(_extract_periods(text, granularity))
    missing = [item for item in expected if item not in actual]
    extra = sorted(actual - set(expected))
    return {
        "start_date": start_date,
        "end_date": end_date,
        "granularity": granularity,
        "calendar": calendar,
        "holidays": sorted(holiday_set),
        "expected_count": len(expected),
        "actual_count": len(actual & set(expected)),
        "missing_items": missing,
        "extra_items": extra,
        "complete": not missing,
    }
