import tempfile
import unittest
from pathlib import Path

import pandas as pd

from er_data import (
    REQUIRED_DATA_COLUMNS,
    SHIFT_LABELS,
    filter_dashboard_data,
    load_dashboard_data,
)


def sample_data() -> pd.DataFrame:
    rows = []
    for index, hour in enumerate([0, 5, 6, 13, 14, 21, 22, 23, 24, -1, 6.5, None]):
        rows.append(
            {
                "encounter_id": index + 1,
                "arrival_year": 2025,
                "arrival_month": 1,
                "arrival_day": 1 + index,
                "arrival_hour": hour,
                "city": "Ankara" if index % 2 == 0 else "Izmir",
                "latitude": 39.9,
                "longitude": 32.8,
                "hospital_type": "Public",
                "triage_level": 3,
                "triage_label": "Level 3",
                "disposition": "Discharged Home",
                "age": 30,
                "ed_los_minutes": 60,
                "adverse_outcome": 0,
                "readmission_30d": 0,
                "primary_icd10_code": "J10",
                "triage_to_physician_min": 15,
                "gender": "F",
                "arrival_mode": "Walk-in",
            }
        )
    return pd.DataFrame(rows)


class DashboardDataTests(unittest.TestCase):
    def write_csv(self, frame: pd.DataFrame) -> Path:
        temporary_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        temporary_file.close()
        path = Path(temporary_file.name)
        self.addCleanup(path.unlink, missing_ok=True)
        frame.to_csv(path, index=False)
        return path

    def test_missing_required_column_fails_with_actionable_message(self) -> None:
        frame = sample_data().drop(columns="encounter_id")

        with self.assertRaisesRegex(ValueError, "encounter_id"):
            load_dashboard_data(self.write_csv(frame))

    def test_invalid_hours_remain_unclassified(self) -> None:
        data = load_dashboard_data(self.write_csv(sample_data()))

        self.assertEqual(
            data["arrival_shift"].astype("string").tolist(),
            [
                SHIFT_LABELS[2],
                SHIFT_LABELS[2],
                SHIFT_LABELS[0],
                SHIFT_LABELS[0],
                SHIFT_LABELS[1],
                SHIFT_LABELS[1],
                SHIFT_LABELS[2],
                SHIFT_LABELS[2],
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
            ],
        )

    def test_date_range_and_optional_filters_are_inclusive(self) -> None:
        data = load_dashboard_data(self.write_csv(sample_data()))

        filtered = filter_dashboard_data(
            data,
            "2025-01-03",
            "2025-01-04",
            cities=["Izmir"],
            shifts=[SHIFT_LABELS[0]],
        )

        self.assertEqual(filtered["encounter_id"].astype(int).tolist(), [4])

    def test_optional_filters_on_absent_column_return_no_rows(self) -> None:
        data = load_dashboard_data(self.write_csv(sample_data().drop(columns="gender")))

        filtered = filter_dashboard_data(
            data,
            "2025-01-01",
            "2025-01-12",
            genders=["F"],
        )

        self.assertTrue(filtered.empty)

    def test_required_schema_contains_encounter_identifier(self) -> None:
        self.assertIn("encounter_id", REQUIRED_DATA_COLUMNS)


if __name__ == "__main__":
    unittest.main()
