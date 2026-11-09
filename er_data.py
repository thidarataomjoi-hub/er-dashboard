from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd


REQUIRED_DATA_COLUMNS = frozenset(
    {
        "encounter_id",
        "arrival_year",
        "arrival_month",
        "arrival_day",
        "arrival_hour",
        "city",
        "latitude",
        "longitude",
        "hospital_type",
        "triage_level",
        "triage_label",
        "disposition",
        "age",
        "ed_los_minutes",
        "adverse_outcome",
        "readmission_30d",
        "primary_icd10_code",
        "triage_to_physician_min",
    }
)

CLOUD_DATA_COLUMNS = (
    "encounter_id",
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "arrival_hour",
    "city",
    "latitude",
    "longitude",
    "hospital_type",
    "triage_level",
    "triage_label",
    "disposition",
    "age",
    "age_group",
    "ed_los_minutes",
    "adverse_outcome",
    "readmission_30d",
    "primary_icd10_code",
    "primary_diagnosis",
    "triage_to_physician_min",
    "gender",
    "arrival_mode",
    "news2_score",
)

NUMERIC_COLUMNS = (
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "arrival_hour",
    "triage_level",
    "age",
    "latitude",
    "longitude",
    "ed_los_minutes",
    "adverse_outcome",
    "readmission_30d",
    "triage_to_physician_min",
)

CATEGORICAL_COLUMNS = (
    "encounter_id",
    "city",
    "hospital_type",
    "triage_label",
    "disposition",
    "age_group",
    "primary_icd10_code",
    "primary_diagnosis",
    "gender",
    "arrival_mode",
)

SHIFT_LABELS = [
    "Morning (06:00–13:59)",
    "Afternoon (14:00–21:59)",
    "Night (22:00–05:59)",
]


def load_dashboard_data(
    file_path: str | Path,
    usecols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    headers = pd.read_csv(file_path, nrows=0).columns
    missing_columns = sorted(REQUIRED_DATA_COLUMNS.difference(headers))
    if missing_columns:
        raise ValueError(f"CSV ขาดคอลัมน์ที่จำเป็น: {', '.join(missing_columns)}")

    selected_columns = (
        [column for column in usecols if column in headers] if usecols else None
    )
    data = pd.read_csv(
        file_path,
        dtype_backend="pyarrow",
        low_memory=False,
        usecols=selected_columns,
    )
    for column in CATEGORICAL_COLUMNS:
        if column in data.columns:
            data[column] = data[column].astype("category")
    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data["arrival_date"] = pd.to_datetime(
        {
            "year": data["arrival_year"],
            "month": data["arrival_month"],
            "day": data["arrival_day"],
        },
        errors="coerce",
    )
    if data["arrival_date"].isna().all():
        raise ValueError(
            "ไม่สามารถอ่านวันที่จากคอลัมน์ arrival_year/month/day ได้"
        )

    hours = data["arrival_hour"].astype("float64")
    valid_hours = hours.between(0, 23) & hours.mod(1).eq(0)
    shifts = pd.Series(pd.NA, index=data.index, dtype="string")
    shifts.loc[valid_hours & hours.between(6, 13)] = SHIFT_LABELS[0]
    shifts.loc[valid_hours & hours.between(14, 21)] = SHIFT_LABELS[1]
    shifts.loc[valid_hours & (hours.ge(22) | hours.le(5))] = SHIFT_LABELS[2]
    data["arrival_shift"] = pd.Categorical(
        shifts,
        categories=SHIFT_LABELS,
        ordered=True,
    )
    return data


def filter_dashboard_data(
    data: pd.DataFrame,
    date_start: date | str | pd.Timestamp,
    date_end: date | str | pd.Timestamp,
    *,
    cities: Sequence[object] | None = None,
    hospitals: Sequence[object] | None = None,
    triage_levels: Sequence[object] | None = None,
    shifts: Sequence[object] | None = None,
    diagnosis_codes: Sequence[object] | None = None,
    dispositions: Sequence[object] | None = None,
    genders: Sequence[object] | None = None,
    arrival_modes: Sequence[object] | None = None,
) -> pd.DataFrame:
    start = pd.Timestamp(date_start)
    end = pd.Timestamp(date_end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    filtered = data[data["arrival_date"].between(start, end)]

    for column, selected in (
        ("city", cities),
        ("hospital_type", hospitals),
        ("triage_level", triage_levels),
        ("arrival_shift", shifts),
        ("primary_icd10_code", diagnosis_codes),
        ("disposition", dispositions),
        ("gender", genders),
        ("arrival_mode", arrival_modes),
    ):
        if selected:
            if column not in filtered.columns:
                return filtered.iloc[0:0]
            values = (
                filtered[column].astype(str)
                if column
                in {"city", "hospital_type", "primary_icd10_code", "gender", "arrival_mode"}
                else filtered[column]
            )
            filtered = filtered[values.isin(selected)]
    return filtered
