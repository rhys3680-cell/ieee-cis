"""FDS 운영 콘솔 프로토타입 (Streamlit).

서비스 계층이 실제로 화면을 지탱하는지 확인하기 위한 임시 UI 다.
최종 UI 는 PySide6 로 만들며 이 파일은 그때 버린다. services/ 는
UI 비의존이므로 그대로 재사용된다.

실행: uv run streamlit run app/dashboard.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from ieee_cis.ml.features import SPLIT_DAY
from ieee_cis.ml.train import MODEL_VERSION
from ieee_cis.services import alerts, cases, metrics, query

st.set_page_config(page_title="FDS 운영 콘솔", layout="wide")

REFRESH_SEC = 1.5


@st.cache_resource
def get_connection():
    """커넥션은 세션 간 재사용한다. 매번 열면 느리다."""
    return query.connect()


@st.cache_data
def get_daily(_con) -> pd.DataFrame:
    """일자별 집계는 365 행뿐이라 한 번 가져와 캐시한다.

    슬라이더를 움직이거나 재생이 돌아도 재쿼리가 없다.
    """
    return metrics.daily(_con)


con = get_connection()
daily = get_daily(con)
day_min, day_max = metrics.day_range(con)

# ---------------------------------------------------------------- 상태
if "day" not in st.session_state:
    st.session_state.day = day_min
if "playing" not in st.session_state:
    st.session_state.playing = False

# ---------------------------------------------------------------- 사이드바
st.sidebar.title("알람 필터")

score_range = st.sidebar.slider("위험도", 0.0, 1.0, (0.5, 1.0), 0.01)
products = st.sidebar.multiselect("상품코드", ["W", "C", "R", "H", "S"])
min_amount = st.sidebar.number_input("최소 금액", min_value=0.0, value=0.0, step=10.0)

st.sidebar.divider()
st.sidebar.caption(
    f"모델 {MODEL_VERSION}\n\n"
    f"학습 구간 1~{SPLIT_DAY}일차\n\n"
    f"검증 구간 {SPLIT_DAY + 1}~182일차\n\n"
    f"라벨 없음 213~395일차"
)

# ---------------------------------------------------------------- 헤더
st.title("FDS 운영 콘솔")

# ---------------------------------------------------------------- 일자 이동
nav = st.columns([1, 1, 1, 1, 6])

if nav[0].button("◀", disabled=st.session_state.day <= day_min, help="이전 일차"):
    st.session_state.day -= 1
    st.session_state.playing = False
if nav[1].button("▶", disabled=st.session_state.day >= day_max, help="다음 일차"):
    st.session_state.day += 1
    st.session_state.playing = False

if st.session_state.playing:
    if nav[2].button("⏸ 정지"):
        st.session_state.playing = False
else:
    if nav[2].button("▶ 재생", disabled=st.session_state.day >= day_max):
        st.session_state.playing = True

if nav[3].button("↺ 처음"):
    st.session_state.day = day_min
    st.session_state.playing = False

day = nav[4].slider(
    "일차", day_min, day_max, st.session_state.day, label_visibility="collapsed"
)
if day != st.session_state.day:
    st.session_state.day = day
    st.session_state.playing = False
day = st.session_state.day

today = daily[daily.day_index == day]
has_data = not today.empty
split_label = today.iloc[0]["dataset_split"] if has_data else "—"

# ---------------------------------------------------------------- KPI
filters = alerts.AlertFilters(
    min_score=score_range[0],
    max_score=score_range[1],
    product_cd=tuple(products),
    min_amount=min_amount or None,
    day=day,
)
stats = alerts.queue_stats(con, filters)

k = st.columns(6)
k[0].metric("일차", f"{day}", help=f"구분: {split_label}")
k[1].metric("거래 건수", f"{int(today.iloc[0]['txn_count']):,}" if has_data else "0")
k[2].metric("알람", f"{stats['total']:,}", help="현재 필터 기준")
k[3].metric("긴급 (0.9+)", f"{stats['critical']:,}")
k[4].metric(
    "거래액",
    f"${today.iloc[0]['total_amount']:,.0f}" if has_data else "—",
)
if stats["labeled"]:
    k[5].metric(
        "사기율",
        f"{stats['fraud_rate']:.1%}",
        help=f"라벨 있는 {stats['labeled']:,}건 기준",
    )
else:
    k[5].metric("사기율", "—", help="이 구간은 라벨이 없습니다")

if not has_data:
    st.info(
        f"{day}일차에는 거래가 없습니다. "
        f"train(1~182일차)과 test(213~395일차) 사이에 30일 공백이 있습니다."
    )

# ---------------------------------------------------------------- 시계열
base = daily.copy()
base["구분"] = base["dataset_split"]

line = (
    alt.Chart(base)
    .mark_line(size=1.5)
    .encode(
        x=alt.X("day_index:Q", title="일차"),
        y=alt.Y("alert_count:Q", title="알람 건수 (위험도 0.5+)"),
        color=alt.Color("구분:N", scale=alt.Scale(scheme="tableau10")),
        tooltip=["day_index", "구분", "txn_count", "alert_count", "critical_count"],
    )
)
marker = (
    alt.Chart(pd.DataFrame({"day_index": [day]}))
    .mark_rule(color="crimson", size=2)
    .encode(x="day_index:Q")
)
st.altair_chart(line + marker, width="stretch")

with st.expander("사기율 추이 (라벨 있는 구간만)"):
    labeled = daily[daily.labeled > 0]
    rate_line = (
        alt.Chart(labeled)
        .mark_line(size=1.5, color="#d62728")
        .encode(
            x=alt.X("day_index:Q", title="일차"),
            y=alt.Y("fraud_rate:Q", title="사기율", axis=alt.Axis(format="%")),
            tooltip=["day_index", "labeled", "known_frauds"],
        )
    )
    st.altair_chart(rate_line + marker, width="stretch")
    st.caption(
        f"test 구간(213~395일차)은 isFraud 가 없어 표시하지 않습니다. "
        f"검증 구간({SPLIT_DAY + 1}~182일차)만 모델이 보지 않은 데이터입니다."
    )

# ---------------------------------------------------------------- 알람큐
st.subheader(f"{day}일차 알람큐")

if day <= SPLIT_DAY:
    st.warning(
        f"학습 구간(1~{SPLIT_DAY}일차)입니다. 모델이 이미 본 데이터라 "
        f"위험도가 실제보다 정확하게 나옵니다. 실제 성능은 "
        f"{SPLIT_DAY + 1}일차 이후에서 판단하세요.",
        icon="⚠️",
    )

page = alerts.fetch_page(con, filters, limit=200)

if page.empty:
    st.info("조건에 맞는 알람이 없습니다.")
else:
    event = st.dataframe(
        page.rename(columns=alerts.QUEUE_COLUMNS),
        width="stretch",
        hide_index=True,
        height=320,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "위험도": st.column_config.ProgressColumn(
                "위험도", min_value=0.0, max_value=1.0, format="%.4f"
            ),
            "금액": st.column_config.NumberColumn("금액", format="$%.2f"),
        },
    )

    # ------------------------------------------------------------ 케이스 상세
    selected = event.selection.rows if event and event.selection else []
    if selected:
        txn_id = int(page.iloc[selected[0]]["TransactionID"])
        detail = cases.get_transaction(con, txn_id)

        st.divider()
        st.subheader(f"케이스 심사 · 거래 {txn_id}")

        d = st.columns(4)
        d[0].metric("위험도", f"{detail['score']:.4f}")
        d[1].metric("금액", f"${detail['TransactionAmt']:,.2f}")
        d[2].metric("구분", detail["dataset_split"])
        label = detail["isFraud"]
        d[3].metric("실제 사기", "—" if pd.isna(label) else ("예" if label else "아니오"))

        tabs = st.tabs(["거래 정보", "카드 이력", "동일 카드 거래"])

        with tabs[0]:
            cols = st.columns(len(cases.DETAIL_GROUPS))
            for col, (group, fields) in zip(cols, cases.DETAIL_GROUPS.items()):
                with col:
                    st.markdown(f"**{group}**")
                    for key, label_ko in fields.items():
                        value = detail.get(key)
                        shown = "—" if value is None or pd.isna(value) else value
                        st.text(f"{label_ko}: {shown}")

        with tabs[1]:
            summary = cases.card_summary(con, txn_id)
            if not summary:
                st.info("카드 이력이 없습니다.")
            else:
                s = st.columns(4)
                s[0].metric("총 거래", f"{summary['txn_count']:,}건")
                s[1].metric("평균 금액", f"${summary['avg_amount']:,.2f}")
                frauds = summary["known_frauds"]
                s[2].metric(
                    "과거 사기",
                    "—" if frauds is None else f"{frauds:,}건",
                    help=(
                        f"라벨 있는 {summary['labeled']:,}건 기준"
                        if summary["labeled"]
                        else "이 카드는 라벨 있는 거래가 없습니다"
                    ),
                )
                rate = summary["fraud_rate"]
                s[3].metric("카드 사기율", "—" if rate is None else f"{rate:.2%}")
                st.caption(
                    f"{summary['first_day']}일차 ~ {summary['last_day']}일차 사용"
                )

        with tabs[2]:
            related = cases.get_related(con, txn_id, limit=20)
            if related.empty:
                st.info("같은 카드의 다른 거래가 없습니다.")
            else:
                st.dataframe(
                    related.rename(columns=alerts.QUEUE_COLUMNS),
                    width="stretch",
                    hide_index=True,
                )

# ---------------------------------------------------------------- 자동 재생
if st.session_state.playing:
    if st.session_state.day >= day_max:
        st.session_state.playing = False
    else:
        import time

        time.sleep(REFRESH_SEC)
        st.session_state.day += 1
        st.rerun()