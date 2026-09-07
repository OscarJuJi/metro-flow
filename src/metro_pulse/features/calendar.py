"""Calendar effects that drive Metro demand.

Three of them matter and none is available in the ridership file itself:

* the **weekday**, which splits demand into commuting days, Saturdays and
  Sundays -- by far the strongest signal in the series;
* **public holidays**, which turn a Tuesday into a Sunday-shaped day; and
* **payday** (``quincena``): the 15th and the last day of the month, plus the
  day after, when trips to shops and banks lift ridership.

Two periods that empty the network are deliberately included even though they
are not statutory holidays in Mexico: **Holy Week**, when schools close and the
city empties, and the **year-end break** between Christmas Eve and New Year's
Day. Leaving them out makes every model overpredict for two weeks a year.
"""

from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd
from dateutil.easter import easter

COUNTRY = "MX"


@lru_cache(maxsize=8)
def _holiday_calendar(first_year: int, last_year: int) -> holidays.HolidayBase:
    return holidays.country_holidays(COUNTRY, years=range(first_year, last_year + 1))


def holiday_names(dates: pd.DatetimeIndex) -> pd.Series:
    """Official Mexican public holiday for each date, empty string when none."""
    calendar = _holiday_calendar(int(dates.year.min()), int(dates.year.max()))
    return pd.Series([calendar.get(d.date(), "") for d in dates], index=dates, dtype="object")


def is_payday(dates: pd.DatetimeIndex) -> pd.Series:
    """The 15th and the last day of the month: Mexican payroll runs twice a month."""
    is_month_end = dates.day == dates.days_in_month
    return pd.Series((dates.day == 15) | is_month_end, index=dates)


@lru_cache(maxsize=8)
def _holy_week_days(first_year: int, last_year: int) -> frozenset:
    """Palm Sunday through Easter Sunday, for every year in the range."""
    days = set()
    for year in range(first_year, last_year + 1):
        easter_sunday = pd.Timestamp(easter(year))
        days.update(pd.date_range(easter_sunday - pd.Timedelta(days=7), easter_sunday))
    return frozenset(days)


def is_holy_week(dates: pd.DatetimeIndex) -> pd.Series:
    """Palm Sunday to Easter Sunday: not a statutory holiday, but the city empties."""
    days = _holy_week_days(int(dates.year.min()), int(dates.year.max()))
    return pd.Series([d in days for d in dates], index=dates)


def is_year_end_break(dates: pd.DatetimeIndex) -> pd.Series:
    """24 December to 1 January, when schools and most offices are shut."""
    return pd.Series(
        ((dates.month == 12) & (dates.day >= 24)) | ((dates.month == 1) & (dates.day == 1)),
        index=dates,
    )


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the calendar block for a set of dates, indexed by date."""
    dates = pd.DatetimeIndex(dates)
    names = holiday_names(dates)
    payday = is_payday(dates)

    features = pd.DataFrame(index=dates)
    features["day_of_week"] = dates.dayofweek
    features["day_of_month"] = dates.day
    features["month"] = dates.month
    features["year"] = dates.year
    features["week_of_year"] = dates.isocalendar().week.to_numpy()
    features["is_weekend"] = dates.dayofweek >= 5
    features["is_holiday"] = names.ne("").to_numpy()
    features["holiday_name"] = names.to_numpy()
    features["is_payday"] = payday.to_numpy()
    features["is_day_after_payday"] = np.roll(payday.to_numpy(), 1)
    features.loc[features.index[0], "is_day_after_payday"] = False

    features["is_holy_week"] = is_holy_week(dates).to_numpy()
    features["is_year_end_break"] = is_year_end_break(dates).to_numpy()

    # A holiday behaves like a Sunday; giving the model one column that already
    # says so beats making it learn the interaction from a sparse dummy.
    sunday_like = features["is_holiday"] | features["is_year_end_break"]
    effective = features["day_of_week"].where(~sunday_like, 6)
    features["effective_day_of_week"] = effective.astype("int8")
    return features
