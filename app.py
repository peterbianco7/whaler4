import io
import re

import pandas as pd
import streamlit as st

st.set_page_config(page_title="WHALER", page_icon="🐋", layout="wide")

BRAND = {
    "blue": "#2F80ED",
    "blue2": "#56A0FF",
    "green": "#27AE60",
    "aqua": "#2DDAE3",
    "bg": "#07131f",
    "bg2": "#0b1b2b",
    "card": "rgba(255,255,255,0.05)",
    "stroke": "rgba(255,255,255,0.10)",
    "text": "#F5FAFF",
    "muted": "rgba(255,255,255,0.70)",
}

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at top right, rgba(47,128,237,0.16), transparent 28%),
            linear-gradient(180deg, {BRAND['bg']} 0%, {BRAND['bg2']} 100%);
        color: {BRAND['text']};
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}
    h1, h2, h3, h4, p, label, div {{ color: {BRAND['text']}; }}
    .hero {{
        background: linear-gradient(135deg, rgba(47,128,237,0.18), rgba(45,218,227,0.10));
        border: 1px solid {BRAND['stroke']};
        padding: 28px;
        border-radius: 22px;
        margin-bottom: 18px;
    }}
    .hero h1 {{ font-size: 3rem; margin: 0 0 8px 0; }}
    .hero p {{ color: {BRAND['muted']}; font-size: 1.05rem; margin: 0; }}
    .card {{
        background: {BRAND['card']};
        border: 1px solid {BRAND['stroke']};
        border-radius: 22px;
        padding: 18px 18px 8px 18px;
        margin-bottom: 18px;
        backdrop-filter: blur(10px);
    }}
    .small-note {{ color: {BRAND['muted']}; font-size: 0.92rem; }}
    .cta {{
        background: linear-gradient(135deg, rgba(47,128,237,0.18), rgba(39,174,96,0.12));
        border: 1px solid {BRAND['stroke']};
        padding: 18px;
        border-radius: 18px;
        text-align: center;
        margin-top: 8px;
    }}
    .metric-label {{ color: {BRAND['muted']}; font-size: 0.88rem; }}
    .metric-value {{ font-size: 2rem; font-weight: 700; line-height: 1.1; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def money_fmt_2(x: float) -> str:
    return f"${x:,.2f}" if pd.notna(x) else "$0.00"


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_money(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return 0.0
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("(", "-").replace(")", "")
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(match.group()) if match else 0.0


def first_existing(df: pd.DataFrame, candidates):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def infer_date_col(df: pd.DataFrame):
    preferred = [
        "date", "transaction_date", "created_at", "created", "timestamp",
        "time", "datetime", "processed_at"
    ]
    col = first_existing(df, preferred)
    if col:
        return col
    for c in df.columns:
        series = pd.to_datetime(df[c], errors="coerce")
        if series.notna().mean() > 0.6:
            return c
    return None


def infer_customer_col(df: pd.DataFrame):
    preferred = [
        "customer", "customer_name", "user", "user_name", "username", "from_user",
        "sender", "name", "member_name", "phrend", "contact", "display_name"
    ]
    col = first_existing(df, preferred)
    if col:
        return col
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["customer", "user", "name", "sender", "member", "phrend"]):
            return c
    return None


def infer_description_col(df: pd.DataFrame):
    preferred = [
        "description", "details", "memo", "note", "type", "transaction_type",
        "event", "activity", "category"
    ]
    col = first_existing(df, preferred)
    if col:
        return col
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["description", "detail", "memo", "note", "type", "event", "category"]):
            return c
    return None


def infer_amount_cols(df: pd.DataFrame):
    credit_col = first_existing(df, ["credits", "credit", "earned", "earnings", "gross", "income"])
    debit_col = first_existing(df, ["debits", "debit", "fees", "fee", "refund", "chargeback"])
    amount_col = first_existing(df, ["amount", "net", "value", "total"])
    return credit_col, debit_col, amount_col


def categorize(row) -> str:
    text = " ".join(
        normalize_text(row.get(col, ""))
        for col in ["description_raw", "type_raw", "category_raw"]
    ).lower()
    if any(k in text for k in ["video", "call", "cam", "phone"]):
        return "Video"
    if any(k in text for k in ["chat", "message", "text", "sext"]):
        return "Chat"
    if any(k in text for k in ["gift", "tip", "rose", "present"]):
        return "Gifts"
    return "Other"


def load_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    df = None

    for encoding in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=encoding)
            break
        except Exception:
            pass

    if df is None:
        raise ValueError("Could not read this CSV.")

    df.columns = [clean_col(c) for c in df.columns]
    df = df.dropna(how="all")

    date_col = infer_date_col(df)
    customer_col = infer_customer_col(df)
    description_col = infer_description_col(df)
    credit_col, debit_col, amount_col = infer_amount_cols(df)

    if date_col is None:
        raise ValueError("I couldn't find a usable date column.")

    work = df.copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce")
    work = work[work["date"].notna()].copy()

    if work.empty:
        raise ValueError("No valid transaction dates found.")

    if customer_col:
        work["customer"] = work[customer_col].astype(str).str.strip()
    else:
        work["customer"] = "Unknown"

    work["customer"] = work["customer"].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

    work["description_raw"] = work[description_col].astype(str).fillna("") if description_col else ""
    work["type_raw"] = work.get("transaction_type", "")
    work["category_raw"] = work.get("category", "")

    if credit_col:
        work["credits"] = work[credit_col].apply(parse_money)
    else:
        work["credits"] = 0.0

    if debit_col:
        work["debits"] = work[debit_col].apply(parse_money)
    else:
        work["debits"] = 0.0

    if not credit_col and amount_col:
        amt = work[amount_col].apply(parse_money)
        work["credits"] = amt.clip(lower=0)
        work["debits"] = (-amt).clip(lower=0)

    work["amount"] = work["credits"] - work["debits"]
    work["type"] = work.apply(categorize, axis=1)
    work["day"] = work["date"].dt.date

    # Dedupe rule: Date + Description + Credits + Debits
    work["dedupe_key"] = (
        work["day"].astype(str)
        + "|" + work["description_raw"].astype(str).str.strip().str.lower()
        + "|" + work["credits"].round(2).astype(str)
        + "|" + work["debits"].round(2).astype(str)
    )
    work = work.drop_duplicates(subset="dedupe_key", keep="first").copy()

    work = work[work["amount"] > 0].copy()

    if work.empty:
        raise ValueError("No positive earnings rows were found after cleaning.")

    return work


def metric_card(label: str, value: str):
    st.markdown(
        f"<div class='card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def build_whale_table(df: pd.DataFrame) -> pd.DataFrame:
    whales = (
        df.groupby("customer", dropna=False)
        .agg(total_earnings=("amount", "sum"), transactions=("amount", "size"))
        .sort_values("total_earnings", ascending=False)
        .reset_index()
    )
    whales.index = whales.index + 1
    whales["rank"] = whales.index
    return whales


def blur_name(name: str) -> str:
    name = str(name)
    if len(name) <= 2:
        return "•" * len(name)
    return name[:1] + "•" * max(3, len(name) - 2) + name[-1:]


def build_display_table(whales: pd.DataFrame) -> pd.DataFrame:
    top10 = whales.head(10).copy()
    display = top10[["rank", "customer", "total_earnings"]].copy()
    display.columns = ["#", "Whale", "Spend"]
    locked = display["#"] >= 4
    display.loc[locked, "Whale"] = display.loc[locked, "Whale"].apply(blur_name)
    display.loc[locked, "Spend"] = "Locked"
    display.loc[~locked, "Spend"] = display.loc[~locked, "Spend"].apply(money_fmt_2)
    return display


def build_top3_breakdown(df: pd.DataFrame, whales: pd.DataFrame) -> pd.DataFrame:
    top3_names = whales.head(3)["customer"].tolist()
    breakdown = (
        df[df["customer"].isin(top3_names)]
        .pivot_table(index="customer", columns="type", values="amount", aggfunc="sum", fill_value=0)
        .reindex(top3_names)
        .fillna(0)
    )
    for col in ["Chat", "Video", "Gifts", "Other"]:
        if col not in breakdown.columns:
            breakdown[col] = 0.0
    return breakdown[["Chat", "Video", "Gifts", "Other"]]


def build_mix(df: pd.DataFrame) -> pd.DataFrame:
    mix = df.groupby("type")["amount"].sum().reindex(["Chat", "Video", "Gifts", "Other"]).fillna(0)
    return mix.to_frame(name="Amount")


def projection_values(df: pd.DataFrame):
    day_min = df["day"].min()
    day_max = df["day"].max()
    days_span = max((pd.to_datetime(day_max) - pd.to_datetime(day_min)).days + 1, 1)
    total = float(df["amount"].sum())
    daily_avg = total / days_span
    monthly = daily_avg * 30
    yearly = daily_avg * 365
    return days_span, daily_avg, monthly, yearly


st.markdown(
    """
    <div class='hero'>
        <h1>WHALER</h1>
        <p>Upload your earnings report. Instantly see who’s really paying you.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Drop in your earnings CSV", type=["csv"])

if not uploaded:
    st.markdown(
        "<div class='card'><p class='small-note'>V1 free shows your core numbers, top whales, top 3 breakdown, and simple projections. No setup. No clutter. Just clarity.</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

try:
    data = load_csv(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

whales = build_whale_table(data)
days_span, daily_avg, monthly_proj, yearly_proj = projection_values(data)
total_earnings = float(data["amount"].sum())
total_whales = int(data["customer"].nunique())
total_calls = int((data["type"] == "Video").sum())
total_chats = int((data["type"] == "Chat").sum())

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card("Total Earnings This Period", money_fmt_2(total_earnings))
with c2:
    metric_card("Total Whales", f"{total_whales:,}")
with c3:
    metric_card("Total Calls", f"{total_calls:,}")
with c4:
    metric_card("Total Chats", f"{total_chats:,}")
with c5:
    metric_card("Monthly Projection", money_fmt_2(monthly_proj))
with c6:
    metric_card("Yearly Projection", money_fmt_2(yearly_proj))

left, right = st.columns([1.1, 0.9])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Top 10 Whales")
    st.caption(f"Based on {days_span} day(s) of deduped earnings data.")
    st.dataframe(build_display_table(whales), use_container_width=True, hide_index=True)
    st.markdown(
        "<div class='cta'><strong>Unlock Whaler Plus</strong><br/>See every whale, unblurred.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Revenue Mix")
    mix_df = build_mix(data)
    st.bar_chart(mix_df, height=290)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Top 3 Whale Breakdown")
st.caption("Chat, video, gifts, and other — broken down by your top 3 spenders.")
stack_df = build_top3_breakdown(data, whales)
st.bar_chart(stack_df, height=360)

pretty_top3 = stack_df.copy()
pretty_top3["Total"] = pretty_top3.sum(axis=1)
for col in pretty_top3.columns:
    pretty_top3[col] = pretty_top3[col].map(money_fmt_2)
st.dataframe(pretty_top3, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Data cleaning notes"):
    st.write("Rows are deduplicated using Date + Description + Credits + Debits.")
    st.write("Only positive earnings rows are included in totals.")
    st.write("Type labels are inferred from your CSV text and grouped into Chat, Video, Gifts, and Other.")
