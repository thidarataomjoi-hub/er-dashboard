import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from er_data import (
    CLOUD_DATA_COLUMNS,
    SHIFT_LABELS,
    filter_dashboard_data,
    load_dashboard_data,
)
from er_security import escape_html


st.set_page_config(
    page_title="DATA Dashboard | ER TURKEY",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
LOCAL_CSV = APP_DIR / "cleaned_train_supp.csv"
LOCAL_CSV_GZ = APP_DIR / "cleaned_train_supp.csv.gz"
TRIAGE_LABELS = {
    1: "Level 1 · วิกฤต",
    2: "Level 2 · เร่งด่วนมาก",
    3: "Level 3 · เร่งด่วน",
    4: "Level 4 · ไม่เร่งด่วน",
    5: "Level 5 · ทั่วไป",
}
TRIAGE_COLORS = {
    1: "#B42318",
    2: "#C2410C",
    3: "#A16207",
    4: "#047857",
    5: "#1D4ED8",
}
TRIAGE_MARKERS = {
    1: "🟥",
    2: "🟧",
    3: "🟨",
    4: "🟩",
    5: "🟦",
}
TRIAGE_COLOR_DOMAIN = [TRIAGE_LABELS[level] for level in TRIAGE_LABELS]
TRIAGE_COLOR_RANGE = [TRIAGE_COLORS[level] for level in TRIAGE_COLORS]
COLOR_PRIMARY = "#079BD5"
COLOR_SUCCESS = "#12B886"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#E05252"
COLOR_PURPLE = "#8B69E8"
COLOR_NEUTRAL = "#94A3B8"
COLOR_INK = "#172F4A"
COLOR_MUTED = "#5D6E80"
COLOR_BORDER = "#E4EBF3"
COLOR_WARNING_DARK = "#B45309"
COLOR_WARNING_BACKGROUND = "#FFF2D8"
DISPOSITION_LABELS = {
    "Discharged Home": "กลับบ้าน",
    "Admitted to Ward": "รับไว้รักษาใน Ward",
    "Admitted to ICU": "รับไว้รักษาใน ICU",
    "Died in ED": "เสียชีวิตใน ER",
    "Transferred Out": "ส่งต่อ",
    "Left AMA": "ออกจากโรงพยาบาลโดยไม่สมัครใจรักษาต่อ",
}
CHART_COLORS = [
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_DANGER,
    COLOR_PURPLE,
]


@st.cache_resource(show_spinner="กำลังโหลดข้อมูลจาก CSV...")
def load_data(
    file_path: str,
    modified_time: float,
    usecols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    del modified_time
    return load_dashboard_data(file_path, usecols=usecols)


def resolve_data_file() -> Path | None:
    if LOCAL_CSV.is_file():
        return LOCAL_CSV
    if LOCAL_CSV_GZ.is_file():
        return LOCAL_CSV_GZ

    data_url = os.environ.get("ER_DATA_URL", "").strip()
    if not data_url:
        try:
            data_url = str(st.secrets.get("ER_DATA_URL", "")).strip()
        except StreamlitSecretNotFoundError:
            data_url = ""
    if not data_url:
        return None

    if urlsplit(data_url).scheme.lower() != "https":
        raise RuntimeError("ER_DATA_URL must use HTTPS.")

    url_hash = hashlib.sha256(data_url.encode("utf-8")).hexdigest()[:16]
    cached_file = Path(tempfile.gettempdir()) / f"er_dashboard_{url_hash}.csv.gz"
    if cached_file.is_file():
        return cached_file

    partial_file = cached_file.with_name(f"{cached_file.name}.part")
    try:
        with urlopen(data_url, timeout=60) as response:
            with partial_file.open("wb") as downloaded:
                shutil.copyfileobj(response, downloaded, length=8 * 1024 * 1024)
        partial_file.replace(cached_file)
    except (OSError, URLError) as exc:
        partial_file.unlink(missing_ok=True)
        raise RuntimeError(
            "ดาวน์โหลด ER_DATA_URL ไม่สำเร็จ กรุณาตรวจสอบ HTTPS URL และสิทธิ์เข้าถึง"
        ) from exc
    return cached_file


def humanize(column: str) -> str:
    return column.replace("_", " ").strip().title()


def set_filter_selection(state_key: str, values: list) -> None:
    st.session_state[state_key] = values


def triage_display_label(level: object, fallback: object = None) -> str:
    try:
        return TRIAGE_LABELS.get(
            int(level), str(fallback if fallback is not None else level)
        )
    except (TypeError, ValueError):
        return str(fallback if fallback is not None else level)


def triage_color_scale() -> alt.Scale:
    return alt.Scale(domain=TRIAGE_COLOR_DOMAIN, range=TRIAGE_COLOR_RANGE)


def make_donut(
    frame: pd.DataFrame,
    name: str,
    title: str,
    color_scale: alt.Scale | None = None,
    legend_columns: int = 2,
) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_arc(innerRadius=58, outerRadius=90, cornerRadius=4)
        .encode(
            theta=alt.Theta("count:Q", stack=True),
            color=alt.Color(
                f"{name}:N",
                scale=(
                    color_scale
                    if color_scale is not None
                    else alt.Scale(range=CHART_COLORS)
                ),
                legend=alt.Legend(
                    title=None,
                    orient="bottom",
                    columns=legend_columns,
                    columnPadding=14,
                    rowPadding=5,
                    labelLimit=160,
                    symbolType="circle",
                ),
            ),
            tooltip=[
                alt.Tooltip(f"{name}:N", title=title),
                alt.Tooltip("count:Q", title="จำนวน", format=","),
            ],
        )
        .properties(height=280)
    )


st.markdown(
    """
    <style>
    :root {
      color-scheme: light;
      --navy: #203653;
      --blue: #079BD5;
      --ink: #1b2d42;
      --muted: #5d6e80;
      --line: #e4ebf3;
      --canvas: #f1f5fa;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
      background: var(--canvas) !important;
      color: var(--ink) !important;
    }
    [data-testid="stHeader"] {
      display: flex !important;
      background: transparent !important;
      height: 2.75rem;
      position: sticky;
      top: 0;
      z-index: 1001;
    }
    [data-testid="stHeader"] button {
      color: #203653 !important;
      background: #fff !important;
      border: 1px solid #d5deea !important;
      border-radius: 8px !important;
    }
    [data-testid="stHeader"] button:hover {
      background: #e8f2fa !important;
    }
    button[aria-label="Collapse sidebar"],
    button[aria-label="Expand sidebar"] {
      visibility: visible !important;
      opacity: 1 !important;
      pointer-events: auto !important;
    }
    [data-testid="stSidebar"] {
      background: var(--navy) !important;
      border-right: 1px solid #d5deea;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
      color: #eaf3fc !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
      color: #a9bfd5 !important;
    }
    [data-testid="stSidebar"] hr { border-color: #3a506b; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div {
      background: #fff !important;
      border: 1px solid #91a8c0 !important;
      border-radius: 8px;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="input"] input {
      color: #203653 !important;
      -webkit-text-fill-color: #203653 !important;
      opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-testid="stDateInput"] input,
    [data-testid="stSidebar"] [data-testid="stDateInput"] input:disabled {
      color: #203653 !important;
      -webkit-text-fill-color: #203653 !important;
      opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-testid="stDateInput"] [data-baseweb="input"] > div {
      background: #fff !important;
      border-color: #91a8c0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stDateInput"] input,
    [data-testid="stSidebar"] [data-testid="stDateInput"] input:disabled,
    [data-testid="stSidebar"] [data-testid="stDateInput"] input::placeholder {
      color: #203653 !important;
      -webkit-text-fill-color: #203653 !important;
      opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stDateInput"] button svg {
      fill: #203653 !important;
      color: #203653 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] input::placeholder {
      color: #203653 !important;
      -webkit-text-fill-color: #203653 !important;
      opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] {
      background: #176ca5 !important;
      border: 1px solid #125a8b !important;
      border-radius: 5px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] *,
    [data-testid="stSidebar"] [data-baseweb="tag"] span {
      color: #fff !important;
      font-weight: 600;
    }
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] [role="option"] {
      background: #fff !important;
      color: #203653 !important;
    }
    [data-baseweb="popover"] [role="option"]:hover {
      background: #e8f2fa !important;
      color: #123b5c !important;
    }
    [data-testid="stSidebar"] .st-key-sidebar-navigation [data-testid="stButton"] {
      margin: 0 0 .35rem;
    }
    [data-testid="stSidebar"] .st-key-sidebar-navigation [data-testid="stButton"] > button {
      width: 100%;
      height: 2.65rem;
      min-height: 2.65rem;
      max-height: 2.65rem;
      justify-content: flex-start;
      text-align: left;
      border: 1px solid transparent;
      background: transparent;
      color: #eaf3fc;
      padding: .55rem .65rem;
    }
    [data-testid="stSidebar"] .st-key-sidebar-navigation [data-testid="stButton"] > button > div {
      width: 100%;
      justify-content: flex-start;
      text-align: left;
    }
    [data-testid="stSidebar"] .st-key-sidebar-navigation [data-testid="stButton"] > button:hover {
      background: #2b4260;
      border-color: #48617e;
      color: white;
    }
    [data-testid="stSidebar"] .st-key-sidebar-navigation [data-testid="stButton"] > button[kind="primary"] {
      background: #0c638f;
      border-color: #1b95c5;
      color: white;
    }
    .block-container { max-width: 1680px; padding: 1rem 1.4rem 2rem; }
    h1, h2, h3, p, label { color: var(--ink); }
    h1 { font-size: 1.55rem !important; letter-spacing: -.025em; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: 1rem !important; }
    [data-testid="stMetric"] {
      background: #fff;
      border: 1px solid var(--line);
      border-left: 4px solid var(--blue);
      border-radius: 12px;
      min-height: 112px;
      padding: 15px 17px;
      box-shadow: 0 3px 12px rgba(29, 55, 80, .045);
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .82rem; }
    [data-testid="stMetricValue"] {
      color: #172f4a !important; font-weight: 750; font-size: 1.35rem !important;
    }
    [data-testid="stTabs"] button { color: var(--muted) !important; }
    .brand {
      color: white; font-size: 1.24rem; font-weight: 750;
      letter-spacing: -.02em; padding: 0 0 4px;
    }
    .brand-small { color: #b7c9dc; font-size: .75rem; letter-spacing: .12em; }
    .topbar {
      display:flex; align-items:center; justify-content:space-between;
      background:#203653; color:white; padding:10px 18px; border-radius:12px;
      margin-bottom:18px; box-shadow:0 3px 10px rgba(18,43,69,.12);
    }
    .topbar strong { color:white; font-size:.96rem; }
    .topbar span { color:#bcd2e5; font-size:.78rem; }
    .page-note { color:var(--muted); font-size:.84rem; margin-top:-.55rem; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; }
    @media (max-width: 640px) {
      .block-container { padding: .75rem .75rem 1.5rem; }
      .topbar { flex-wrap: wrap; gap: .35rem; padding: 10px 12px; }
      .topbar span { font-size: .72rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    DATA_FILE = resolve_data_file()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if DATA_FILE is None:
    st.error(
        "ไม่พบไฟล์ CSV กรุณาวาง cleaned_train_supp.csv หรือ "
        "cleaned_train_supp.csv.gz ไว้ข้าง er_dashboard.py "
        "หรือกำหนด ER_DATA_URL ใน Streamlit Secrets"
    )
    st.stop()

REMOTE_DATA = DATA_FILE not in (LOCAL_CSV, LOCAL_CSV_GZ)

try:
    data = load_data(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime,
        usecols=CLOUD_DATA_COLUMNS if REMOTE_DATA else None,
    )
except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
    st.error(f"อ่านไฟล์ CSV ไม่สำเร็จ: {exc}")
    st.stop()

minimum_date = data["arrival_date"].min()
maximum_date = data["arrival_date"].max()
min_date = minimum_date.date()
max_date = maximum_date.date()
cities = sorted(data["city"].dropna().astype(str).unique().tolist())
hospitals = sorted(data["hospital_type"].dropna().astype(str).unique().tolist())
triage_levels = sorted(data["triage_level"].dropna().unique().tolist())
diagnosis_codes = sorted(data["primary_icd10_code"].dropna().astype(str).unique().tolist())
genders = (
    sorted(data["gender"].dropna().astype(str).unique().tolist())
    if "gender" in data.columns
    else []
)
arrival_modes = (
    sorted(data["arrival_mode"].dropna().astype(str).unique().tolist())
    if "arrival_mode" in data.columns
    else []
)
dispositions = sorted(data["disposition"].dropna().astype(str).unique().tolist())

with st.sidebar:
    st.markdown(
        '<div class="brand">🏥 DATA Dashboard</div>'
        '<div class="brand-small">EMERGENCY CARE · CSV ANALYTICS</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("ภาพรวม & การวิเคราะห์")
    pages = [
        ("หน้าแรก (ER Dashboard)", "📊  หน้าแรก (ER Dashboard)"),
        ("แผนที่บริการ", "📍  แผนที่บริการ"),
        ("วิเคราะห์การวินิจฉัย", "🔎  ค้นหาผู้ป่วยตามโรค (ICD-10)"),
        ("เวลารอคอย", "⏱️  วิเคราะห์เวลารอคอย"),
        ("ข้อมูลคลินิก", "🩺  ข้อมูลคลินิก"),
        ("สำรวจข้อมูลทั้งหมด", "🗂️  สำรวจข้อมูลทั้งหมด"),
    ]
    if "active_page" not in st.session_state:
        st.session_state.active_page = "หน้าแรก (ER Dashboard)"
    with st.container(key="sidebar-navigation"):
        for page_value, page_label in pages:
            if st.button(
                page_label,
                key=f"nav_{page_value}",
                type="primary"
                if st.session_state.active_page == page_value
                else "secondary",
                width="stretch",
            ):
                st.session_state.active_page = page_value
                st.rerun()
    page = st.session_state.active_page
    st.divider()
    st.markdown("#### กรองข้อมูล")
    date_preset = st.selectbox(
        "ช่วงข้อมูล",
        ["วันล่าสุด", "30 วันล่าสุด", "ข้อมูลทั้งหมด", "กำหนดช่วงวันที่"],
        index=2,
    )
    if date_preset == "วันล่าสุด":
        date_start = date_end = max_date
    elif date_preset == "30 วันล่าสุด":
        date_start = max(min_date, max_date - pd.Timedelta(days=29).to_pytimedelta())
        date_end = max_date
    elif date_preset == "ข้อมูลทั้งหมด":
        date_start, date_end = min_date, max_date
    else:
        selected_range = st.date_input(
            "เลือกช่วงวันที่",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY",
        )
        date_start, date_end = (
            selected_range
            if isinstance(selected_range, tuple)
            else (selected_range, selected_range)
        )
    st.session_state.setdefault("filter_cities", [])
    st.session_state.setdefault("filter_hospitals", [])
    st.session_state.setdefault("filter_triage", [])
    st.session_state.setdefault("filter_shifts", [])

    st.markdown("**เมือง**")
    selected_cities = st.multiselect(
        "เลือกเมือง",
        cities,
        key="filter_cities",
        placeholder="พิมพ์ค้นหา หรือเลือกเมือง",
        label_visibility="collapsed",
    )
    select_all, clear = st.columns([1.5, 1])
    select_all.button(
        "เลือกทั้งหมด",
        key="select_all_cities",
        width="stretch",
        on_click=set_filter_selection,
        args=("filter_cities", cities),
    )
    clear.button(
        "ล้าง",
        key="clear_cities",
        width="stretch",
        on_click=set_filter_selection,
        args=("filter_cities", []),
    )

    st.markdown("**ประเภทโรงพยาบาล**")
    selected_hospitals = st.multiselect(
        "เลือกประเภทโรงพยาบาล",
        hospitals,
        key="filter_hospitals",
        placeholder="พิมพ์ค้นหา หรือเลือกประเภท",
        label_visibility="collapsed",
    )
    select_all, clear = st.columns([1.5, 1])
    select_all.button(
        "เลือกทั้งหมด",
        key="select_all_hospitals",
        width="stretch",
        on_click=set_filter_selection,
        args=("filter_hospitals", hospitals),
    )
    clear.button(
        "ล้าง",
        key="clear_hospitals",
        width="stretch",
        on_click=set_filter_selection,
        args=("filter_hospitals", []),
    )

    st.markdown("**ระดับคัดกรอง / ESI**")
    selected_triage = st.multiselect(
        "เลือกระดับคัดกรอง",
        triage_levels,
        key="filter_triage",
        placeholder="พิมพ์ค้นหา หรือเลือกระดับ",
        format_func=lambda level: (
            f"{TRIAGE_MARKERS.get(int(level), '⚪')} "
            f"{TRIAGE_LABELS.get(int(level), f'Level {level}')}"
        ),
        label_visibility="collapsed",
    )
    select_all, clear = st.columns([1.5, 1])
    select_all.button(
        "เลือกทั้งหมด",
        key="select_all_triage",
        width="stretch",
        on_click=set_filter_selection,
        args=("filter_triage", triage_levels),
    )
    clear.button(
        "ล้าง",
        key="clear_triage",
        width="stretch",
        on_click=set_filter_selection,
        args=("filter_triage", []),
    )

    st.markdown("**ช่วงเวลา (Shift)**")
    selected_shifts = st.multiselect(
        "เลือกช่วงเวลา",
        SHIFT_LABELS,
        key="filter_shifts",
        placeholder="เลือกช่วงเวลา",
        label_visibility="collapsed",
    )
    select_all, clear = st.columns([1.5, 1])
    select_all.button(
        "เลือกทั้งหมด",
        key="select_all_shifts",
        width="stretch",
        on_click=set_filter_selection,
        args=("filter_shifts", SHIFT_LABELS),
    )
    clear.button(
        "ล้าง",
        key="clear_shifts",
        width="stretch",
        on_click=set_filter_selection,
        args=("filter_shifts", []),
    )
    with st.expander("ตัวกรองเพิ่มเติม"):
        selected_diagnosis_codes = st.multiselect(
            "รหัส ICD-10", diagnosis_codes, default=[]
        )
        selected_dispositions = st.multiselect(
            "ผลการจำหน่าย", dispositions, default=[]
        )
        selected_genders = (
            st.multiselect("เพศ", genders, default=[]) if genders else []
        )
        selected_arrival_modes = (
            st.multiselect("วิธีเดินทางมา", arrival_modes, default=[])
            if arrival_modes
            else []
        )
    st.caption(
        "Shift: Morning 06–13:59 · Afternoon 14–21:59 · Night 22–05:59"
    )
    st.divider()
    st.caption(
        f"ข้อมูล {len(data):,} แถว · {len(data.columns)} ฟิลด์"
        + (" · โหมด Cloud โหลดเฉพาะฟิลด์หลักเพื่อลด RAM" if REMOTE_DATA else "")
    )

filter_options = {
    "cities": selected_cities,
    "hospitals": selected_hospitals,
    "triage_levels": selected_triage,
    "shifts": selected_shifts,
    "diagnosis_codes": selected_diagnosis_codes,
    "dispositions": selected_dispositions,
    "genders": selected_genders,
    "arrival_modes": selected_arrival_modes,
}
filtered = filter_dashboard_data(data, date_start, date_end, **filter_options)

if page == "หน้าแรก (ER Dashboard)":
    trend_start = date_start.replace(month=1, day=1)
    trend_end = date_end.replace(month=12, day=31)
    trend_data = filter_dashboard_data(
        data,
        trend_start,
        trend_end,
        **filter_options,
    )

st.markdown(
    f"""
    <div class="topbar">
      <strong>DATA Dashboard　›　บริการผู้ป่วย　›　ห้องฉุกเฉิน (ER)</strong>
      <span>{date_start.strftime("%d/%m/%Y")} — {date_end.strftime("%d/%m/%Y")}　 ·  CSV data</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("ไม่พบข้อมูลตามตัวกรองที่เลือก กรุณาปรับช่วงวันที่หรือเงื่อนไข")
    st.stop()

st.title(
    "ข้อมูลสถิติตึกอุบัติเหตุและฉุกเฉิน (ER Dashboard)"
    if page == "หน้าแรก (ER Dashboard)"
    else page
)
st.markdown(
    f'<p class="page-note">ข้อมูลผู้รับบริการ ER ในช่วงที่เลือก · '
    f'{len(filtered):,} จาก {len(data):,} รายการ</p>',
    unsafe_allow_html=True,
)

diagnosis_code = (
    filtered["primary_icd10_code"]
    .astype("string")
    .str.upper()
    .fillna("")
)
injury_mask = diagnosis_code.str.startswith(("S", "T"))
trauma_count = int(injury_mask.sum())
non_trauma_count = int((~injury_mask).sum())
admit_count = int(
    filtered["disposition"].isin(["Admitted to Ward", "Admitted to ICU"]).sum()
)
refer_count = int(filtered["disposition"].eq("Transferred Out").sum())
los_mean = pd.to_numeric(filtered["ed_los_minutes"], errors="coerce").mean()

if page == "หน้าแรก (ER Dashboard)":
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("ผู้รับบริการ ER ทั้งหมด (ราย)", f"{len(filtered):,}")
    kpi2.metric("เจ็บป่วย · Non-Trauma* (ราย)", f"{non_trauma_count:,}")
    kpi3.metric("อุบัติเหตุ · ICD-10 S/T* (ราย)", f"{trauma_count:,}")
    with kpi4:
        st.markdown(
            f"""
            <div style="background:white;border:1px solid {COLOR_BORDER};border-left:4px solid {COLOR_PRIMARY};
            border-radius:12px;padding:14px 18px;min-height:112px;box-shadow:0 3px 12px rgba(29,55,80,.045)">
              <div style="font-size:.82rem;color:{COLOR_MUTED};margin-bottom:12px">ผลการรักษา</div>
              <div style="font-size:1.04rem;color:{COLOR_SUCCESS};font-weight:700">Admit　{admit_count:,} ราย</div>
              <div style="font-size:1.04rem;color:{COLOR_PRIMARY};font-weight:700;margin-top:8px">Refer Out　{refer_count:,} ราย</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "* แบ่งกลุ่มจาก ICD-10 ที่ขึ้นต้นด้วย S/T เท่านั้น ไม่ใช่การจำแนก trauma จากการตรวจสอบเวชระเบียน"
    )

    st.markdown("")
    hourly_column, triage_column = st.columns([1.7, 1])
    with hourly_column:
        with st.container(border=True):
            st.subheader("📈 จำนวนผู้เข้ารับบริการรายชั่วโมง (ER Hourly Flow)")
            hourly = (
                filtered.groupby("arrival_hour", observed=True)
                .size()
                .reindex(range(24), fill_value=0)
                .rename("จำนวนผู้รับบริการ")
                .rename_axis("ชั่วโมง")
                .reset_index()
            )
            hourly["เวลา"] = hourly["ชั่วโมง"].map(lambda hour: f"{int(hour):02d}:00")
            hourly_chart = (
                alt.Chart(hourly)
                .mark_line(
                    point=alt.OverlayMarkDef(filled=True, size=48),
                    interpolate="monotone",
                    strokeWidth=3,
                    color=COLOR_PRIMARY,
                )
                .encode(
                    x=alt.X("เวลา:N", sort=None, axis=alt.Axis(labelAngle=0, title=None)),
                    y=alt.Y("จำนวนผู้รับบริการ:Q", title="จำนวนราย", scale=alt.Scale(zero=True)),
                    tooltip=[
                        alt.Tooltip("เวลา:N", title="เวลา"),
                        alt.Tooltip("จำนวนผู้รับบริการ:Q", title="ผู้รับบริการ", format=","),
                    ],
                )
                .properties(height=310)
            )
            st.altair_chart(hourly_chart, width="stretch")
    with triage_column:
        with st.container(border=True):
            st.subheader("🩺 สัดส่วนระดับคัดกรอง")
            triage_data = (
                filtered.groupby(["triage_level", "triage_label"], observed=True)
                .size()
                .rename("count")
                .reset_index()
            )
            triage_data["ระดับ"] = triage_data.apply(
                lambda row: triage_display_label(
                    row["triage_level"], row["triage_label"]
                ),
                axis=1,
            )
            st.altair_chart(
                make_donut(
                    triage_data,
                    "ระดับ",
                    "ระดับคัดกรอง",
                    color_scale=triage_color_scale(),
                    legend_columns=3,
                ),
                width="stretch",
            )

    outcome_column, monthly_column = st.columns([1, 1.7])
    with outcome_column:
        with st.container(border=True):
            st.subheader("📋 การจำหน่ายผู้ป่วย")
            outcome_data = (
                filtered["disposition"]
                .astype("string")
                .fillna("ไม่ระบุ")
                .value_counts()
                .rename_axis("ผลการจำหน่าย")
                .rename("จำนวน")
                .reset_index()
            )
            outcome_data["สถานะ"] = outcome_data["ผลการจำหน่าย"].map(
                lambda item: DISPOSITION_LABELS.get(str(item), str(item))
            )
            outcome_chart = (
                alt.Chart(outcome_data)
                .mark_bar(cornerRadiusEnd=5)
                .encode(
                    x=alt.X("จำนวน:Q", title="จำนวนราย"),
                    y=alt.Y("สถานะ:N", sort="-x", title=None),
                    color=alt.Color(
                        "สถานะ:N",
                        scale=alt.Scale(range=CHART_COLORS),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("สถานะ:N", title="ผลการจำหน่าย"),
                        alt.Tooltip("จำนวน:Q", title="จำนวน", format=","),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(outcome_chart, width="stretch")
    with monthly_column:
        with st.container(border=True):
            st.subheader("📅 แนวโน้มผู้รับบริการรายเดือน")
            st.caption(
                "แสดงเดือนเต็มของปีที่เลือก เพื่อให้เห็นแนวโน้มแม้ KPI จะอยู่ในช่วงวันเดียว"
            )
            monthly = (
                trend_data.groupby(
                    ["arrival_year", "arrival_month"], observed=True
                )
                .size()
                .rename("ผู้รับบริการ")
                .reset_index()
                .sort_values(["arrival_year", "arrival_month"])
            )
            monthly["เดือน"] = pd.to_datetime(
                {
                    "year": monthly["arrival_year"],
                    "month": monthly["arrival_month"],
                    "day": 1,
                }
            )
            monthly_chart = (
                alt.Chart(monthly)
                .mark_line(
                    point=alt.OverlayMarkDef(filled=True, size=46),
                    interpolate="monotone",
                    strokeWidth=3,
                    color=COLOR_PRIMARY,
                )
                .encode(
                    x=alt.X("เดือน:T", title=None, axis=alt.Axis(format="%b %Y")),
                    y=alt.Y("ผู้รับบริการ:Q", title="จำนวนราย"),
                    tooltip=[
                        alt.Tooltip("เดือน:T", title="เดือน", format="%b %Y"),
                        alt.Tooltip("ผู้รับบริการ:Q", title="จำนวน", format=","),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(monthly_chart, width="stretch")

    st.markdown("")
    small1, small2, small3, small4 = st.columns(4)
    small1.metric(
        "เวลาอยู่ ER เฉลี่ย",
        f"{los_mean:,.1f} นาที" if pd.notna(los_mean) else "—",
    )
    small2.metric("ผลลัพธ์ไม่พึงประสงค์", f"{filtered['adverse_outcome'].mean() * 100:.1f}%")
    small3.metric("กลับมารักษาซ้ำใน 30 วัน", f"{filtered['readmission_30d'].mean() * 100:.1f}%")
    if "news2_score" in filtered.columns:
        news2 = pd.to_numeric(filtered["news2_score"], errors="coerce").mean()
        small4.metric("NEWS2 เฉลี่ย", f"{news2:.2f}" if pd.notna(news2) else "—")
    else:
        small4.metric("เมืองที่มีข้อมูล", f"{filtered['city'].nunique():,} เมือง")

elif page == "แผนที่บริการ":
    st.markdown(
        '<p class="page-note">ขนาดวงกลมแทนจำนวน Visits · คลิกหรือชี้ที่จุดเพื่อดูเมือง '
        "ประเภทโรงพยาบาล และโรคที่พบบ่อย</p>",
        unsafe_allow_html=True,
    )
    geo = filtered.dropna(subset=["latitude", "longitude", "city"]).copy()
    if geo.empty:
        st.info("ไม่มีข้อมูลพิกัดในตัวกรองนี้")
    else:
        city_map = (
            geo.groupby(["city", "latitude", "longitude"], observed=True)
            .agg(
                visits=("encounter_id", "size"),
                adverse_outcomes=("adverse_outcome", "sum"),
                wait_mean=("triage_to_physician_min", "mean"),
            )
            .reset_index()
        )
        city_map["adverse_rate"] = (
            city_map["adverse_outcomes"] / city_map["visits"] * 100
        )

        hospital_summary = (
            geo.groupby(["city", "hospital_type"], observed=True)
            .size()
            .rename("hospital_visits")
            .reset_index()
            .sort_values(["city", "hospital_visits"], ascending=[True, False])
            .drop_duplicates("city")
            .rename(
                columns={
                    "hospital_type": "โรงพยาบาลที่พบมากสุด",
                    "hospital_visits": "Visits ของประเภทโรงพยาบาล",
                }
            )
        )
        diagnosis_summary = (
            geo.groupby(
                ["city", "primary_icd10_code", "primary_diagnosis"],
                observed=True,
                dropna=False,
            )
            .size()
            .rename("diagnosis_visits")
            .reset_index()
            .sort_values(["city", "diagnosis_visits"], ascending=[True, False])
            .drop_duplicates("city")
            .rename(
                columns={
                    "primary_icd10_code": "รหัสโรคเด่น",
                    "primary_diagnosis": "ชื่อโรคเด่น",
                    "diagnosis_visits": "Visits ของโรคเด่น",
                }
            )
        )
        city_map = city_map.merge(hospital_summary, on="city", how="left")
        city_map = city_map.merge(diagnosis_summary, on="city", how="left")
        for source_column, tooltip_column in (
            ("city", "city_tooltip"),
            ("โรงพยาบาลที่พบมากสุด", "hospital_type_tooltip"),
            ("รหัสโรคเด่น", "diagnosis_code_tooltip"),
            ("ชื่อโรคเด่น", "diagnosis_name_tooltip"),
        ):
            city_map[tooltip_column] = (
                city_map[source_column]
                .astype("string")
                .fillna("ไม่ระบุ")
                .map(escape_html)
            )
        city_map["radius"] = (
            6000
            + 24000
            * (city_map["visits"] / city_map["visits"].max()).pow(0.5)
        )
        city_map["city_label"] = ""
        city_map.loc[
            city_map["visits"].nlargest(min(10, len(city_map))).index,
            "city_label",
        ] = city_map.loc[
            city_map["visits"].nlargest(min(10, len(city_map))).index,
            "city",
        ]
        points_layer = pdk.Layer(
            "ScatterplotLayer",
            data=city_map,
            get_position="[longitude, latitude]",
            get_radius="radius",
            get_fill_color=[7, 128, 190, 210],
            get_line_color=[255, 255, 255, 245],
            line_width_min_pixels=2,
            stroked=True,
            pickable=True,
            auto_highlight=True,
        )
        labels_layer = pdk.Layer(
            "TextLayer",
            data=city_map[city_map["city_label"] != ""],
            get_position="[longitude, latitude]",
            get_text="city_label",
            get_size=14,
            get_color=[25, 45, 65, 255],
            get_background_color=[255, 255, 255, 220],
            background=True,
            get_pixel_offset=[0, -22],
            get_alignment_baseline="'bottom'",
            pickable=True,
        )
        map_kpis = st.columns(4)
        map_kpis[0].metric("เมืองที่แสดงบนแผนที่", f"{city_map['city'].nunique():,}")
        map_kpis[1].metric("Visits ในผลกรอง", f"{int(city_map['visits'].sum()):,}")
        map_kpis[2].metric(
            "ผลลัพธ์ไม่พึงประสงค์",
            f"{city_map['adverse_outcomes'].sum() / city_map['visits'].sum() * 100:.1f}%",
        )
        map_kpis[3].metric(
            "เวลารอเฉลี่ย",
            f"{geo['triage_to_physician_min'].mean():.1f} นาที",
        )

        with st.container(border=True):
            st.subheader("📍 แผนที่ ER ตามเมือง")
            st.pydeck_chart(
                pdk.Deck(
                    layers=[points_layer, labels_layer],
                    initial_view_state=pdk.ViewState(
                        latitude=float(city_map["latitude"].mean()),
                        longitude=float(city_map["longitude"].mean()),
                        zoom=5,
                        pitch=0,
                    ),
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    tooltip={
                        "html": (
                            "<b>{city_tooltip}</b><br/>"
                            "ประเภทโรงพยาบาลที่พบมากสุด: {hospital_type_tooltip}<br/>"
                            "Visits ของประเภทนี้: {Visits ของประเภทโรงพยาบาล}<br/>"
                            "จำนวน Visits ทั้งเมือง: {visits}<br/>"
                            "โรคที่พบบ่อย: {diagnosis_code_tooltip} · {diagnosis_name_tooltip}<br/>"
                            "Visits ของโรคเด่น: {Visits ของโรคเด่น}<br/>"
                            "ผลลัพธ์ไม่พึงประสงค์: {adverse_rate}%<br/>"
                            "เวลารอเฉลี่ย: {wait_mean} นาที"
                        ),
                        "style": {
                            "backgroundColor": "#203653",
                            "color": "white",
                            "fontSize": "13px",
                            "padding": "10px",
                        },
                    },
                ),
                height=720,
                width="stretch",
            )
            st.caption(
                "ป้ายชื่อแสดง 10 เมืองที่มี Visits สูงสุด · วงกลมใหญ่หมายถึงจำนวน Visits มาก"
            )

        detail_columns = [
            "city",
            "visits",
            "โรงพยาบาลที่พบมากสุด",
            "Visits ของประเภทโรงพยาบาล",
            "รหัสโรคเด่น",
            "ชื่อโรคเด่น",
            "Visits ของโรคเด่น",
            "adverse_rate",
            "wait_mean",
        ]
        city_details = city_map[detail_columns].sort_values(
            "visits", ascending=False
        )
        city_details = city_details.rename(
            columns={
                "city": "เมือง",
                "visits": "จำนวน Visits",
                "adverse_rate": "ผลลัพธ์ไม่พึงประสงค์ (%)",
                "wait_mean": "เวลารอเฉลี่ย (นาที)",
            }
        )
        with st.container(border=True):
            st.subheader("รายละเอียดโรงพยาบาลและโรคที่พบบ่อยในแต่ละเมือง")
            st.dataframe(
                city_details,
                hide_index=True,
                width="stretch",
                height=min(460, 42 * (len(city_details) + 1)),
                column_config={
                    "จำนวน Visits": st.column_config.NumberColumn(format="%d"),
                    "Visits ของประเภทโรงพยาบาล": st.column_config.NumberColumn(
                        format="%d"
                    ),
                    "Visits ของโรคเด่น": st.column_config.NumberColumn(format="%d"),
                    "ผลลัพธ์ไม่พึงประสงค์ (%)": st.column_config.NumberColumn(
                        format="%.1f%%"
                    ),
                    "เวลารอเฉลี่ย (นาที)": st.column_config.NumberColumn(
                        format="%.1f"
                    ),
                },
            )
    st.info(
        "พิกัดในไฟล์เป็นพิกัดระดับเมืองในประเทศตุรกี ไม่ใช่ตำแหน่งรายบุคคลหรือพิกัดโรงพยาบาล "
        "และ CSV มีเฉพาะประเภทโรงพยาบาล ไม่มีชื่อสถานพยาบาลจริง "
        "ชื่อโรคแสดงจากการวินิจฉัยหลักที่พบบ่อยในเมืองนั้น · แผนที่พื้นหลังต้องใช้อินเทอร์เน็ต"
    )

elif page == "วิเคราะห์การวินิจฉัย":
    st.markdown(
        '<p class="page-note">วิเคราะห์รหัสโรคตามการวินิจฉัยหลัก (ICD-10) '
        "ในช่วงวันที่และตัวกรองที่เลือก</p>",
        unsafe_allow_html=True,
    )
    diagnosis_fields = ["primary_icd10_code"]
    if "primary_diagnosis" in filtered.columns:
        diagnosis_fields.append("primary_diagnosis")
    diagnosis_summary = (
        filtered.groupby(diagnosis_fields, dropna=False, observed=True)
        .agg(
            visits=("encounter_id", "size"),
            adverse_outcomes=("adverse_outcome", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "primary_icd10_code": "รหัสโรค",
                "primary_diagnosis": "ชื่อการวินิจฉัย",
            }
        )
        .sort_values("visits", ascending=False)
        .reset_index(drop=True)
    )
    if "ชื่อการวินิจฉัย" not in diagnosis_summary:
        diagnosis_summary["ชื่อการวินิจฉัย"] = "ไม่ระบุ"
    diagnosis_summary["รหัสโรค"] = (
        diagnosis_summary["รหัสโรค"].astype("string").fillna("ไม่ระบุ")
    )
    diagnosis_summary["ชื่อการวินิจฉัย"] = (
        diagnosis_summary["ชื่อการวินิจฉัย"].astype("string").fillna("ไม่ระบุ")
    )
    diagnosis_summary["อัตราผลลัพธ์ (%)"] = (
        diagnosis_summary["adverse_outcomes"]
        .div(diagnosis_summary["visits"].where(diagnosis_summary["visits"] != 0))
        .fillna(0)
        .mul(100)
    )
    diagnosis_summary["สัดส่วน (%)"] = (
        diagnosis_summary["visits"] / diagnosis_summary["visits"].sum() * 100
    )
    diagnosis_summary["การวินิจฉัย"] = (
        diagnosis_summary["รหัสโรค"].astype(str)
        + " · "
        + diagnosis_summary["ชื่อการวินิจฉัย"].astype(str)
    )

    top_diagnosis = diagnosis_summary.iloc[0]
    distinct_visits = int(filtered["encounter_id"].nunique())
    adverse_total = int(diagnosis_summary["adverse_outcomes"].sum())
    kpi1, kpi2, kpi3, kpi4 = st.columns([1, 1.15, 1.15, 1.35])
    kpi1.metric(
        "จำนวนรหัสโรค (ICD-10)",
        f"{diagnosis_summary['รหัสโรค'].nunique():,}",
        help="จำนวนรหัสวินิจฉัยหลักที่แตกต่างกันในช่วงข้อมูล",
    )
    kpi2.metric(
        "Encounter ID ไม่ซ้ำ",
        f"{distinct_visits:,}",
        help="จำนวนรหัส encounter ที่แตกต่างกัน ไม่ใช่จำนวนผู้ป่วยไม่ซ้ำ เพราะ CSV ไม่มี HN",
    )
    kpi3.metric("จำนวนครั้งรับบริการ (Visits)", f"{len(filtered):,}")
    with kpi4:
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid {COLOR_BORDER};border-left:4px solid {COLOR_WARNING};
            border-radius:12px;padding:14px 17px;min-height:112px;
            box-shadow:0 3px 12px rgba(29,55,80,.045)">
              <div style="font-size:.82rem;color:{COLOR_MUTED};margin-bottom:10px">อันดับ 1 ที่พบมากที่สุด</div>
              <div style="font-size:1.04rem;color:{COLOR_INK};font-weight:750">
                <span style="display:inline-block;background:{COLOR_WARNING_BACKGROUND};color:{COLOR_WARNING_DARK};
                padding:3px 8px;border-radius:6px;margin-right:6px">{escape_html(top_diagnosis['รหัสโรค'])}</span>
                {escape_html(top_diagnosis['ชื่อการวินิจฉัย'])}
              </div>
              <div style="font-size:.78rem;color:{COLOR_MUTED};margin-top:8px">
                {int(top_diagnosis['visits']):,} ครั้งในช่วงข้อมูลที่เลือก
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        f"นับ encounter ID ที่ไม่ซ้ำได้ {distinct_visits:,} รายการ · "
        f"พบผลลัพธ์ไม่พึงประสงค์ {adverse_total:,} รายการ · "
        "ข้อมูลไม่มี HN จึงไม่สามารถยืนยันจำนวนผู้ป่วยรายบุคคลได้"
    )

    chart_column, donut_column = st.columns([1.75, 1])
    top_twenty = diagnosis_summary.head(20).copy()
    with chart_column:
        with st.container(border=True):
            st.subheader("📊 20 อันดับรหัสโรค (Top 20 ICD-10)")
            st.caption(
                "เปรียบเทียบจำนวนครั้งรับบริการกับจำนวน encounter ที่มีผลลัพธ์ไม่พึงประสงค์"
            )
            chart_base = pd.DataFrame(
                {
                    "รหัสโรค": top_twenty["รหัสโรค"].astype(str).to_numpy(dtype=object),
                    "ชื่อการวินิจฉัย": top_twenty["ชื่อการวินิจฉัย"]
                    .astype(str)
                    .to_numpy(dtype=object),
                    "visits": pd.to_numeric(
                        top_twenty["visits"], errors="coerce"
                    ).fillna(0).to_numpy(dtype="int64"),
                    "adverse_outcomes": pd.to_numeric(
                        top_twenty["adverse_outcomes"], errors="coerce"
                    ).fillna(0).to_numpy(dtype="int64"),
                    "อัตราผลลัพธ์ (%)": pd.to_numeric(
                        top_twenty["อัตราผลลัพธ์ (%)"], errors="coerce"
                    ).fillna(0).to_numpy(dtype="float64"),
                    "สัดส่วน (%)": pd.to_numeric(
                        top_twenty["สัดส่วน (%)"], errors="coerce"
                    ).fillna(0).to_numpy(dtype="float64"),
                }
            )
            visits_chart = chart_base.copy()
            visits_chart["ประเภท"] = "จำนวนครั้ง (Visits)"
            visits_chart["จำนวน"] = visits_chart["visits"]
            outcomes_chart = chart_base.copy()
            outcomes_chart["ประเภท"] = "ผลลัพธ์ไม่พึงประสงค์"
            outcomes_chart["จำนวน"] = outcomes_chart["adverse_outcomes"]
            chart_data = pd.concat(
                [visits_chart, outcomes_chart], ignore_index=True
            )
            diagnosis_chart = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X(
                        "รหัสโรค:N",
                        sort=top_twenty["รหัสโรค"].tolist(),
                        title=None,
                        axis=alt.Axis(labelAngle=0, labelLimit=75),
                    ),
                    xOffset=alt.XOffset("ประเภท:N"),
                    y=alt.Y("จำนวน:Q", title="จำนวนรายการ", axis=alt.Axis(format="~s")),
                    color=alt.Color(
                        "ประเภท:N",
                        scale=alt.Scale(
                            domain=[
                                "จำนวนครั้ง (Visits)",
                                "ผลลัพธ์ไม่พึงประสงค์",
                            ],
                            range=CHART_COLORS[:2],
                        ),
                        legend=alt.Legend(orient="top", title=None),
                    ),
                    tooltip=[
                        alt.Tooltip("รหัสโรค:N", title="ICD-10"),
                        alt.Tooltip("ชื่อการวินิจฉัย:N", title="การวินิจฉัย"),
                        alt.Tooltip("ประเภท:N", title="ตัวชี้วัด"),
                        alt.Tooltip("จำนวน:Q", title="ค่าของแท่ง", format=",.0f"),
                        alt.Tooltip("visits:Q", title="Visits ทั้งหมด", format=",.0f"),
                        alt.Tooltip("สัดส่วน (%):Q", title="สัดส่วน Visits", format=".2f"),
                        alt.Tooltip(
                            "adverse_outcomes:Q",
                            title="ผลลัพธ์ไม่พึงประสงค์",
                            format=",.0f",
                        ),
                        alt.Tooltip(
                            "อัตราผลลัพธ์ (%):Q",
                            title="อัตราผลลัพธ์ไม่พึงประสงค์",
                            format=".2f",
                        ),
                    ],
                )
                .properties(height=315)
            )
            st.altair_chart(diagnosis_chart, width="stretch")
    with donut_column:
        with st.container(border=True):
            st.subheader("🥧 สัดส่วน Top 5 รหัสโรค")
            st.caption("รหัสอื่น ๆ รวมเป็นหมวด “อื่น ๆ”")
            donut_data = top_twenty.head(5).copy()
            donut_data["กลุ่มโรค"] = (
                donut_data["รหัสโรค"].astype(str)
                + " · "
                + donut_data["ชื่อการวินิจฉัย"].astype(str)
            )
            donut_data = donut_data[["กลุ่มโรค", "visits", "สัดส่วน (%)"]].rename(
                columns={"visits": "count"}
            )
            other_visits = int(
                diagnosis_summary.iloc[5:]["visits"].sum()
            )
            if other_visits:
                donut_data.loc[len(donut_data)] = [
                    f"อื่น ๆ ({max(0, len(diagnosis_summary) - 5)} รหัส)",
                    other_visits,
                    other_visits / diagnosis_summary["visits"].sum() * 100,
                ]
            donut = (
                alt.Chart(donut_data)
                .mark_arc(innerRadius=76, outerRadius=122, cornerRadius=4)
                .encode(
                    theta=alt.Theta("count:Q", stack=True),
                    color=alt.Color(
                        "กลุ่มโรค:N",
                        scale=alt.Scale(
                            range=CHART_COLORS + [COLOR_NEUTRAL]
                        ),
                        legend=alt.Legend(
                            orient="bottom",
                            columns=2,
                            title=None,
                            labelLimit=130,
                            symbolType="square",
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("กลุ่มโรค:N", title="รหัสโรค"),
                        alt.Tooltip("count:Q", title="จำนวนครั้ง", format=","),
                        alt.Tooltip("สัดส่วน (%):Q", title="สัดส่วน Visits", format=".2f"),
                    ],
                )
                .properties(height=365)
            )
            st.altair_chart(donut, width="stretch")

    with st.container(border=True):
        heading, controls = st.columns([1.2, 1.8], vertical_alignment="center")
        with heading:
            st.subheader("☷ ตารางจัดอันดับโรค")
            st.caption("ค้นหารหัสหรือชื่อการวินิจฉัย · ดาวน์โหลดผลลัพธ์ได้")
        with controls:
            control1, control2, control3 = st.columns([1.5, 0.8, 0.8])
            search_term = control1.text_input(
                "ค้นหา ICD-10 / ชื่อโรค",
                placeholder="พิมพ์รหัสหรือชื่อ...",
                label_visibility="collapsed",
                key="diagnosis_search",
            )
            top_limit = control2.selectbox(
                "แสดง",
                [10, 20, 50, "ทั้งหมด"],
                index=0,
                format_func=lambda value: (
                    "ทั้งหมด" if value == "ทั้งหมด" else f"Top {value}"
                ),
                label_visibility="collapsed",
                key="diagnosis_limit",
            )
            download_button = control3

        table_data = diagnosis_summary.copy()
        if search_term:
            search_mask = (
                table_data["รหัสโรค"].str.contains(search_term, case=False, na=False)
                | table_data["ชื่อการวินิจฉัย"].str.contains(
                    search_term, case=False, na=False
                )
            )
            table_data = table_data[search_mask]
        if top_limit != "ทั้งหมด":
            table_data = table_data.head(int(top_limit))
        table_data = table_data.reset_index(drop=True)
        table_data.insert(0, "อันดับ", range(1, len(table_data) + 1))
        display_data = table_data[
            [
                "อันดับ",
                "รหัสโรค",
                "ชื่อการวินิจฉัย",
                "visits",
                "adverse_outcomes",
                "อัตราผลลัพธ์ (%)",
            ]
        ].rename(
            columns={
                "visits": "จำนวนครั้ง (Visits)",
                "adverse_outcomes": "ผลลัพธ์ไม่พึงประสงค์",
            }
        )
        st.dataframe(
            display_data,
            hide_index=True,
            width="stretch",
            height=min(445, 42 * (len(display_data) + 1)),
            column_config={
                "อันดับ": st.column_config.NumberColumn(width="small"),
                "รหัสโรค": st.column_config.TextColumn(width="small"),
                "ชื่อการวินิจฉัย": st.column_config.TextColumn(width="large"),
                "จำนวนครั้ง (Visits)": st.column_config.NumberColumn(
                    format="%d", width="small"
                ),
                "ผลลัพธ์ไม่พึงประสงค์": st.column_config.NumberColumn(
                    format="%d", width="small"
                ),
                "อัตราผลลัพธ์ (%)": st.column_config.ProgressColumn(
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                    width="medium",
                ),
            },
        )
        download_button.download_button(
            "⬇ Excel / CSV",
            data=display_data.to_csv(index=False).encode("utf-8-sig"),
            file_name="er_icd10_ranking.csv",
            mime="text/csv",
            width="stretch",
        )

elif page == "ข้อมูลคลินิก":
    clinical_left, clinical_right = st.columns(2)
    with clinical_left:
        with st.container(border=True):
            st.subheader("การกระจายอายุ")
            ages = pd.to_numeric(filtered["age"], errors="coerce").dropna()
            age_groups = pd.cut(
                ages,
                bins=[-1, 17, 39, 59, 74, float("inf")],
                labels=["0–17", "18–39", "40–59", "60–74", "75+"],
            )
            age_counts = (
                age_groups.value_counts(sort=False)
                .rename_axis("ช่วงอายุ")
                .rename("จำนวน")
                .reset_index()
            )
            st.bar_chart(age_counts.set_index("ช่วงอายุ"), color=COLOR_PRIMARY)
    with clinical_right:
        with st.container(border=True):
            st.subheader("การคัดกรองและผลลัพธ์")
            triage_outcome = (
                filtered.groupby(
                    ["triage_level", "triage_label", "adverse_outcome"],
                    observed=True,
                )
                .size()
                .rename("จำนวน")
                .reset_index()
            )
            triage_outcome["ระดับ"] = triage_outcome.apply(
                lambda row: triage_display_label(
                    row["triage_level"], row["triage_label"]
                ),
                axis=1,
            )
            triage_outcome["ผล"] = triage_outcome["adverse_outcome"].map(
                {0: "ไม่พบผลลัพธ์ไม่พึงประสงค์", 1: "พบผลลัพธ์ไม่พึงประสงค์"}
            )
            chart = (
                alt.Chart(triage_outcome)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "ระดับ:N",
                        title=None,
                        sort=TRIAGE_COLOR_DOMAIN,
                    ),
                    y=alt.Y("จำนวน:Q", title="จำนวนราย"),
                    color=alt.Color(
                        "ระดับ:N",
                        scale=triage_color_scale(),
                        legend=None,
                    ),
                    column=alt.Column("ผล:N", title=None),
                    tooltip=[
                        alt.Tooltip("ระดับ:N", title="ระดับคัดกรอง"),
                        alt.Tooltip("ผล:N", title="ผลลัพธ์"),
                        alt.Tooltip("จำนวน:Q", format=","),
                    ],
                )
                .properties(height=330)
            )
            st.altair_chart(chart, width="stretch")
    st.subheader("ตัวชี้วัดตามกลุ่ม")
    numeric_fields = [
        column
        for column in filtered.columns
        if pd.api.types.is_numeric_dtype(filtered[column])
        and column not in {"encounter_id", "latitude", "longitude"}
    ]
    categorical_fields = [
        column
        for column in (
            "triage_label",
            "disposition",
            "gender",
            "age_group",
            "arrival_mode",
            "hospital_type",
            "city",
        )
        if column in filtered.columns
    ]
    if numeric_fields and categorical_fields:
        measure = st.selectbox(
            "ตัวชี้วัด",
            numeric_fields,
            index=numeric_fields.index("ed_los_minutes")
            if "ed_los_minutes" in numeric_fields
            else 0,
            format_func=humanize,
        )
        grouping = st.selectbox(
            "แบ่งกลุ่มตาม",
            categorical_fields,
            index=categorical_fields.index("triage_label"),
            format_func=humanize,
        )
        grouped = (
            filtered.groupby(grouping, observed=True)[measure]
            .mean()
            .sort_values(ascending=False)
            .head(25)
            .rename(f"ค่าเฉลี่ย {humanize(measure)}")
        )
        if grouping == "triage_label":
            triage_by_label = (
                filtered[["triage_level", "triage_label"]]
                .drop_duplicates()
                .set_index("triage_label")["triage_level"]
            )
            grouped_chart = grouped.rename_axis("ระดับ").reset_index()
            grouped_chart["ระดับ"] = grouped_chart["ระดับ"].map(
                lambda label: triage_display_label(
                    triage_by_label.get(label), label
                )
            )
            st.altair_chart(
                alt.Chart(grouped_chart)
                .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                .encode(
                    x=alt.X("ระดับ:N", sort=TRIAGE_COLOR_DOMAIN, title=None),
                    y=alt.Y(grouped.name + ":Q", title=grouped.name),
                    color=alt.Color(
                        "ระดับ:N",
                        scale=triage_color_scale(),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("ระดับ:N", title="ระดับคัดกรอง"),
                        alt.Tooltip(grouped.name + ":Q", title=grouped.name, format=".1f"),
                    ],
                ),
                width="stretch",
            )
        else:
            st.bar_chart(grouped, color=COLOR_PRIMARY)

elif page == "เวลารอคอย":
    st.markdown(
        '<p class="page-note">เวลาตั้งแต่คัดกรองจนถึงพบแพทย์ '
        "คำนวณจาก triage_to_physician_min ใน CSV</p>",
        unsafe_allow_html=True,
    )
    waits = pd.to_numeric(filtered["triage_to_physician_min"], errors="coerce")
    waits = waits[waits.notna() & waits.ge(0)]
    if waits.empty:
        st.info("ไม่มีข้อมูลเวลารอในตัวกรองที่เลือก")
    else:
        threshold = st.slider(
            "เกณฑ์เวลารอนาน (นาที)",
            min_value=5,
            max_value=120,
            value=30,
            step=5,
            help="ใช้กำหนด KPI จำนวนและสัดส่วน visits ที่รอมากกว่าเกณฑ์นี้",
        )
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Visits ที่มีเวลารอ", f"{len(waits):,}")
        kpi2.metric("เวลารอเฉลี่ย", f"{waits.mean():,.1f} นาที")
        kpi3.metric("เวลารอมัธยฐาน", f"{waits.median():,.1f} นาที")
        long_wait_count = int(waits.gt(threshold).sum())
        kpi4.metric(
            f"รอเกิน {threshold} นาที",
            f"{long_wait_count:,} ({long_wait_count / len(waits) * 100:.1f}%)",
        )

        wait_records = filtered.loc[waits.index].assign(wait_min=waits)
        shift_summary = (
            wait_records.groupby("arrival_shift", observed=True)
            .agg(
                wait_mean=("wait_min", "mean"),
                wait_median=("wait_min", "median"),
                wait_p90=("wait_min", lambda values: values.quantile(0.9)),
                visits=("wait_min", "size"),
            )
            .reset_index()
        )
        triage_summary = (
            wait_records.groupby(["triage_level", "triage_label"], observed=True)
            .agg(
                wait_mean=("wait_min", "mean"),
                wait_median=("wait_min", "median"),
                visits=("wait_min", "size"),
            )
            .reset_index()
        )
        triage_summary["ระดับ"] = triage_summary.apply(
            lambda row: triage_display_label(
                row["triage_level"], row["triage_label"]
            ),
            axis=1,
        )

        with st.container(border=True):
            st.subheader("จุดที่ใช้เวลารอนานที่สุด และเหตุผลที่ข้อมูลบอกได้")
            st.write(
                "ไฟล์วัดเวลารอได้เพียงขั้นตอน **คัดกรอง → พบแพทย์** "
                "จึงยังเปรียบเทียบไม่ได้ว่าขั้นตอนอื่น เช่น ตรวจรักษา หรือรอจำหน่าย "
                "ใช้เวลานานกว่าเพียงใด"
            )
            peak_columns = st.columns(2)
            if not shift_summary.empty:
                peak_shift = shift_summary.loc[shift_summary["wait_mean"].idxmax()]
                peak_columns[0].metric(
                    "ช่วงเวลาที่รอเฉลี่ยนานที่สุด",
                    f"{peak_shift['arrival_shift']}",
                    f"{peak_shift['wait_mean']:.1f} นาที · {peak_shift['visits']:,} visits",
                )
            else:
                peak_columns[0].metric("ช่วงเวลาที่รอเฉลี่ยนานที่สุด", "ไม่มีข้อมูล")
            if not triage_summary.empty:
                peak_triage = triage_summary.loc[triage_summary["wait_mean"].idxmax()]
                peak_columns[1].metric(
                    "ระดับคัดกรองที่รอเฉลี่ยนานที่สุด",
                    triage_display_label(
                        peak_triage["triage_level"], peak_triage["triage_label"]
                    ),
                    f"{peak_triage['wait_mean']:.1f} นาที · {peak_triage['visits']:,} visits",
                )
            else:
                peak_columns[1].metric("ระดับคัดกรองที่รอเฉลี่ยนานที่สุด", "ไม่มีข้อมูล")

            if not shift_summary.empty:
                busiest_shift = shift_summary.loc[shift_summary["visits"].idxmax()]
                if busiest_shift["arrival_shift"] == peak_shift["arrival_shift"]:
                    st.write(
                        f"**ข้อสังเกต:** {peak_shift['arrival_shift']} เป็นทั้งช่วงที่เวลารอเฉลี่ยสูงสุด "
                        f"และมี visits มากที่สุด ({peak_shift['visits']:,} ราย) "
                        "ปริมาณผู้มารับบริการอาจเป็นปัจจัยที่ควรตรวจสอบเพิ่มเติม"
                    )
                else:
                    st.write(
                        f"**ข้อสังเกต:** ช่วงที่เวลารอเฉลี่ยสูงสุดคือ {peak_shift['arrival_shift']} "
                        f"แต่ช่วงที่มี visits มากที่สุดคือ {busiest_shift['arrival_shift']} "
                        "ดังนั้นจำนวน visits เพียงอย่างเดียวอธิบายช่วงเวลารอสูงสุดไม่ได้"
                    )
            st.caption(
                "เป็นความสัมพันธ์ที่พบในข้อมูล ไม่ใช่ข้อสรุปเชิงเหตุและผล "
                "ไฟล์นี้ไม่มีข้อมูลจำนวนบุคลากร ความยาวคิว ความพร้อมของแพทย์ "
                "หรือความหนาแน่นผู้ป่วย จึงระบุสาเหตุที่แท้จริงไม่ได้"
            )

        monthly = filtered.loc[waits.index].copy()
        monthly["wait_min"] = waits
        monthly["เดือน"] = monthly["arrival_date"].dt.to_period("M").dt.to_timestamp()
        monthly_summary = (
            monthly.groupby("เดือน", observed=True)
            .agg(
                wait_mean=("wait_min", "mean"),
                wait_median=("wait_min", "median"),
                visits=("wait_min", "size"),
            )
            .reset_index()
        )
        trend_chart = (
            alt.Chart(monthly_summary)
            .mark_line(point=True, strokeWidth=2.5, color=COLOR_PRIMARY)
            .encode(
                x=alt.X("เดือน:T", title="เดือน", axis=alt.Axis(format="%b %Y")),
                y=alt.Y("wait_mean:Q", title="เวลารอเฉลี่ย (นาที)", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("เดือน:T", title="เดือน", format="%B %Y"),
                    alt.Tooltip("wait_mean:Q", title="เฉลี่ย (นาที)", format=".1f"),
                    alt.Tooltip("wait_median:Q", title="มัธยฐาน (นาที)", format=".1f"),
                    alt.Tooltip("visits:Q", title="จำนวน Visits", format=","),
                ],
            )
            .properties(height=300)
        )
        with st.container(border=True):
            st.subheader("แนวโน้มเวลารอรายเดือน")
            st.altair_chart(trend_chart, width="stretch")

        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.subheader("เวลารอแยกตาม Shift")
                shift_chart = (
                    alt.Chart(shift_summary)
                    .mark_bar(
                        cornerRadiusTopLeft=5,
                        cornerRadiusTopRight=5,
                        color=COLOR_SUCCESS,
                    )
                    .encode(
                        x=alt.X("arrival_shift:N", title="ช่วงเวลา", sort=SHIFT_LABELS),
                        y=alt.Y("wait_mean:Q", title="เวลารอเฉลี่ย (นาที)"),
                        tooltip=[
                            alt.Tooltip("arrival_shift:N", title="Shift"),
                            alt.Tooltip("wait_mean:Q", title="เฉลี่ย (นาที)", format=".1f"),
                            alt.Tooltip("wait_median:Q", title="มัธยฐาน (นาที)", format=".1f"),
                            alt.Tooltip("wait_p90:Q", title="เปอร์เซ็นไทล์ 90 (นาที)", format=".1f"),
                            alt.Tooltip("visits:Q", title="จำนวน Visits", format=","),
                        ],
                    )
                    .properties(height=310)
                )
                st.altair_chart(shift_chart, width="stretch")

        with right:
            with st.container(border=True):
                st.subheader("เวลารอตามระดับคัดกรอง")
                triage_chart = (
                    alt.Chart(triage_summary)
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                    .encode(
                        x=alt.X(
                            "ระดับ:N",
                            title="ระดับคัดกรอง (Triage)",
                            sort=TRIAGE_COLOR_DOMAIN,
                        ),
                        y=alt.Y("wait_mean:Q", title="เวลารอเฉลี่ย (นาที)"),
                        color=alt.Color(
                            "ระดับ:N",
                            scale=triage_color_scale(),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("ระดับ:N", title="ระดับคัดกรอง"),
                            alt.Tooltip("wait_mean:Q", title="เฉลี่ย (นาที)", format=".1f"),
                            alt.Tooltip("wait_median:Q", title="มัธยฐาน (นาที)", format=".1f"),
                            alt.Tooltip("visits:Q", title="จำนวน Visits", format=","),
                        ],
                    )
                    .properties(height=310)
                )
                st.altair_chart(triage_chart, width="stretch")

        wait_bands = pd.cut(
            waits,
            bins=[-0.001, 15, 30, 60, 120, float("inf")],
            labels=["0–15", ">15–30", ">30–60", ">60–120", ">120"],
        )
        wait_distribution = (
            wait_bands.value_counts(sort=False)
            .rename_axis("ช่วงเวลารอ (นาที)")
            .rename("visits")
            .reset_index()
        )
        wait_distribution["สัดส่วน (%)"] = (
            wait_distribution["visits"] / wait_distribution["visits"].sum() * 100
        )
        with st.container(border=True):
            st.subheader("การกระจายเวลารอ")
            distribution_chart = (
                alt.Chart(wait_distribution)
                .mark_bar(
                    cornerRadiusTopLeft=4,
                    cornerRadiusTopRight=4,
                    color=COLOR_PURPLE,
                )
                .encode(
                    x=alt.X("ช่วงเวลารอ (นาที):N", title="ช่วงเวลารอ (นาที)", sort=None),
                    y=alt.Y("visits:Q", title="จำนวน Visits"),
                    tooltip=[
                        alt.Tooltip("ช่วงเวลารอ (นาที):N", title="ช่วงเวลารอ"),
                        alt.Tooltip("visits:Q", title="จำนวน Visits", format=","),
                        alt.Tooltip("สัดส่วน (%):Q", title="สัดส่วน Visits", format=".2f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(distribution_chart, width="stretch")

        heatmap_data = (
            filtered.loc[waits.index]
            .assign(wait_min=waits)
            .groupby(["arrival_shift", "triage_level", "triage_label"], observed=True)
            .agg(wait_mean=("wait_min", "mean"), visits=("wait_min", "size"))
            .reset_index()
        )
        heatmap_data["ระดับ"] = heatmap_data.apply(
            lambda row: triage_display_label(
                row["triage_level"], row["triage_label"]
            ),
            axis=1,
        )
        with st.container(border=True):
            st.subheader("Heatmap: ระดับคัดกรอง × Shift")
            heatmap = (
                alt.Chart(heatmap_data)
                .mark_rect(cornerRadius=3)
                .encode(
                    x=alt.X("arrival_shift:N", title="ช่วงเวลา", sort=SHIFT_LABELS),
                    y=alt.Y(
                        "ระดับ:N",
                        title="ระดับคัดกรอง",
                        sort=TRIAGE_COLOR_DOMAIN,
                    ),
                    color=alt.Color(
                        "wait_mean:Q",
                        title="เฉลี่ย (นาที)",
                        scale=alt.Scale(scheme="tealblues"),
                    ),
                    tooltip=[
                        alt.Tooltip("ระดับ:N", title="ระดับคัดกรอง"),
                        alt.Tooltip("arrival_shift:N", title="Shift"),
                        alt.Tooltip("wait_mean:Q", title="เวลารอเฉลี่ย (นาที)", format=".1f"),
                        alt.Tooltip("visits:Q", title="จำนวน Visits", format=","),
                    ],
                )
                .properties(height=290)
            )
            st.altair_chart(heatmap, width="stretch")
        st.caption(
            "Shift: Morning 06–13:59, Afternoon 14–21:59, Night 22–05:59. "
            "ตัวกรองนี้ใช้ระดับคัดกรองใน CSV (Triage Level); ไฟล์ไม่ได้ยืนยันว่าเป็นระบบ ESI "
            "และช่วงเวลารอแบ่งตามนาทีเพื่อสะท้อนข้อมูลจริง"
        )

else:
    st.subheader("สำรวจข้อมูลทั้งหมดจาก CSV")
    st.caption(
        f"ข้อมูลที่กรองแล้ว {len(filtered):,} แถว · "
        f"เลือกดูหรือดาวน์โหลดตัวอย่างจาก {len(data.columns) - 1} ฟิลด์"
    )
    info = pd.DataFrame(
        {
            "คอลัมน์": [column for column in data.columns if column != "arrival_date"],
            "ชนิดข้อมูล": [
                str(data[column].dtype)
                for column in data.columns
                if column != "arrival_date"
            ],
            "ค่าว่าง (%)": [
                (filtered[column].isna().mean() * 100)
                for column in data.columns
                if column != "arrival_date"
            ],
            "จำนวนค่าที่แตกต่าง": [
                filtered[column].nunique(dropna=True)
                for column in data.columns
                if column != "arrival_date"
            ],
        }
    )
    st.dataframe(info, hide_index=True, height=360, width="stretch")
    raw_columns = st.multiselect(
        "เลือกคอลัมน์สำหรับดูข้อมูล",
        [column for column in data.columns if column != "arrival_date"],
        default=[
            column
            for column in (
                "arrival_year",
                "arrival_month",
                "city",
                "age",
                "triage_label",
                "disposition",
                "ed_los_minutes",
            )
            if column in data.columns
        ],
        format_func=humanize,
    )
    row_limit = st.select_slider(
        "จำนวนแถวตัวอย่าง",
        options=[100, 250, 500, 1000, 2000],
        value=500,
    )
    if raw_columns:
        sample = filtered[raw_columns].head(row_limit)
        st.dataframe(sample, hide_index=True, height=420, width="stretch")
        st.download_button(
            "ดาวน์โหลดตัวอย่าง CSV",
            data=sample.to_csv(index=False).encode("utf-8-sig"),
            file_name="er_data_sample.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "สรุปจากไฟล์ CSV ที่แนบ ไม่ใช่ข้อมูล Real-time หรือเครื่องมือวินิจฉัยทางการแพทย์"
)
