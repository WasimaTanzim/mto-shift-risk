"""MTO Road Incident Risk: shift app for field supervisors (US-003, US-004, US-007).

Built for one person: the field supervisor of a maintenance zone, opening the app a few minutes
before a 6-hour shift. The main page answers three questions for that zone only:
how risky is this shift, what should I do, and should I follow it.

Reads the reporting tables in workspace.mto_analytics through a Databricks SQL warehouse and
writes supervisor decisions (follows and overrides) to workspace.mto_analytics.app_override_log.
Local test: set MTO_LOCAL_DIR to a folder of parquet copies of the tables.
"""
import os
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

CATALOG, SCHEMA = os.getenv("MTO_CATALOG", "workspace"), os.getenv("MTO_SCHEMA", "mto_analytics")
T = lambda name: f"{CATALOG}.{SCHEMA}.{name}"
HERE = os.path.dirname(os.path.abspath(__file__))
# Databricks mode when a SQL warehouse is attached; otherwise read the data snapshot shipped in ./data
_DATA = os.path.join(HERE, "data") if os.path.exists(os.path.join(HERE, "data", "snap_predictions.csv")) else HERE
LOCAL = os.getenv("MTO_LOCAL_DIR") or (None if os.getenv("DATABRICKS_WAREHOUSE_ID") else _DATA)
TZ = "America/Toronto"
MIN_PER_KM = 0.8          # from the response-time model: 7.36 + 0.798 x road km
SMALL_GAIN_MIN = 2.0      # a move that saves less than this in the zone is "optional"
NEAR_MIN = 20.0           # High-risk stretches reachable within this already count as covered (median Critical response)

ZONE_NAMES = {"Zone-A-SW": "Zone A (Southwest)", "Zone-B-SE": "Zone B (Southeast)",
              "Zone-C-NE": "Zone C (Northeast)", "Zone-D-NW": "Zone D (Northwest)",
              "Zone-E-Central": "Zone E (Central)"}
SCENARIO_LABEL = {"A": "Clear weather", "B": "Winter storm", "C": "Freezing rain + construction"}

st.set_page_config(page_title="MTO Shift Risk", layout="wide")

# ---------------------------------------------------------------- data access
@st.cache_resource
def connection():
    from databricks import sql
    from databricks.sdk.core import Config
    cfg = Config()                                    # the app's own credentials (service principal)
    return sql.connect(server_hostname=cfg.host,
                       http_path=f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}",
                       credentials_provider=lambda: cfg.authenticate)

def query(sql_text, params=None):
    with connection().cursor() as cur:
        cur.execute(sql_text, params or {})
        return cur.fetchall_arrow().to_pandas()

@st.cache_data
def local(name):
    pq, csv = os.path.join(LOCAL, f"{name}.parquet"), os.path.join(LOCAL, f"{name}.csv")
    df = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(csv)
    for c in ("window_start", "shift_start_utc"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True).dt.tz_localize(None)
    return df

def has_local(name):
    return any(os.path.exists(os.path.join(LOCAL, f"{name}.{e}")) for e in ("parquet", "csv"))

def utc(ts):
    """Any timestamp (naive UTC, or tz-aware from SQL) -> naive UTC."""
    ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts

def aware(ts):
    return utc(ts).tz_localize("UTC").to_pydatetime()

def local_time(ts):
    return utc(ts).tz_localize("UTC").tz_convert(TZ)

def shift_label(ts):
    s = local_time(ts); e = s + pd.Timedelta(hours=6)
    return f"{s:%a %d %b %Y}, {s:%H:%M}–{e:%H:%M}"

COLS = ("segment_id, maintenance_zone, highway_number, risk_xgboost, risk_level, risk_level_order, "
        "confidence_band, confidence_text, reason_1, reason_2, reason_1_group, recommended_action")

@st.cache_data(ttl=3600)
def shift_rows(window_start, zone):
    if LOCAL:
        p = local("snap_predictions" if has_local("snap_predictions") else "dash_predictions_2023")
        p = p[(p.window_start == utc(window_start)) & (p.maintenance_zone == zone)]
        return p[[c.strip() for c in COLS.split(",")]].reset_index(drop=True)
    return query(f"SELECT {COLS} FROM {T('dash_predictions_2023')} WHERE window_start = :w AND maintenance_zone = :z",
                 {"w": aware(window_start), "z": zone})

@st.cache_data(ttl=3600)
def track_record():
    """How often each risk level / confidence turned into a Critical incident in 2023."""
    if LOCAL and has_local("snap_track_record"):
        return local("snap_track_record")
    if LOCAL:
        p = local("dash_predictions_2023")
        g = p.groupby(["risk_level", "confidence_band"]).critical_incident.mean().reset_index(name="hit_rate")
        return pd.concat([g, p.groupby("risk_level").critical_incident.mean().reset_index(name="hit_rate").assign(confidence_band="All")])
    return query(f"""SELECT risk_level, COALESCE(confidence_band, 'All') AS confidence_band, AVG(critical_incident) AS hit_rate
                     FROM {T('dash_predictions_2023')} GROUP BY GROUPING SETS ((risk_level, confidence_band), (risk_level))""")

@st.cache_data(ttl=3600)
def patrol(name):
    return local(name) if LOCAL else query(f"SELECT * FROM {T(name)}")

def log_decision(row):
    if LOCAL:
        f = os.path.join(LOCAL, "app_override_log.csv")
        pd.DataFrame([row]).to_csv(f, mode="a", header=not os.path.exists(f), index=False)
        return
    query(f"""INSERT INTO {T('app_override_log')}
              (override_id, logged_at_utc, supervisor_id, window_start_utc, segment_id, maintenance_zone,
               model_risk_level, model_risk, recommended_action, supervisor_decision, reason, note)
              VALUES (:override_id, :logged_at_utc, :supervisor_id, :window_start_utc, :segment_id, :maintenance_zone,
               :model_risk_level, :model_risk, :recommended_action, :supervisor_decision, :reason, :note)""", row)

def read_log():
    if LOCAL:
        f = os.path.join(LOCAL, "app_override_log.csv")
        return pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
    return query(f"SELECT * FROM {T('app_override_log')} ORDER BY logged_at_utc DESC LIMIT 500")

# ---------------------------------------------------------------- plain-language helpers
def stretch(seg):
    """SEG-401-0002 -> 'Highway 401, stretch 2'."""
    parts = str(seg).split("-")
    if len(parts) < 3:
        return str(seg)
    road = parts[1] if parts[1].isalpha() else f"Highway {parts[1]}"
    return f"{road}, stretch {int(parts[2])}"

def depot(code):
    return str(code).replace("DEP-", "Depot ")

def zname(z):
    return ZONE_NAMES.get(z, z)

def reasons(r):
    out = [r.reason_1] + ([r.reason_2] if isinstance(r.reason_2, str) and r.reason_2 not in ("", "None") else [])
    return "; ".join(x[0].lower() + x[1:] for x in out if isinstance(x, str))

def hit_rate(level, conf="All"):
    tr = track_record()
    v = tr[(tr.risk_level == level) & (tr.confidence_band == conf)].hit_rate
    return float(v.iloc[0]) if len(v) else None

def one_in(p):
    return f"about 1 in {max(1, round(1 / p))}" if p and p > 0 else "rarely"

# ---------------------------------------------------------------- sidebar
st.sidebar.title("MTO Shift Risk")
zone = st.sidebar.selectbox("Your zone", list(ZONE_NAMES), index=None, format_func=zname, placeholder="Choose your zone")
supervisor = st.sidebar.text_input("Your supervisor ID", placeholder="e.g. SUP-D-014")
page = st.sidebar.radio("Go to", ["My shift", "Decision log", "What-if planning"],
                        captions=["What to do before this shift", "Moves confirmed or declined", "Scenarios for planners"])
st.sidebar.caption("Prototype. Runs on sample conditions until connected to the Ministry's live feeds. Times are Eastern.")

# ---------------------------------------------------------------- page 1: my shift
if page == "My shift":
    if zone is None:
        st.header("Before your shift")
        st.info("Choose your zone on the left to see your shift.")
        st.stop()

    zp, mv = patrol("dash_patrol_zone_plan"), patrol("dash_patrol_moves")
    zp = zp.assign(shift=zp.shift_start_utc.map(utc))
    planned = (zp[zp.case == "All depots open"].drop_duplicates("scenario")
                 .set_index("shift").scenario.to_dict())             # shift -> scenario name
    by_label = {SCENARIO_LABEL.get(str(sc)[:1], str(sc)): s for s, sc in planned.items()}
    order = [l for l in ["Winter storm", "Freezing rain + construction", "Clear weather"] if l in by_label]
    order += [l for l in by_label if l not in order]

    st.header(f"Your next shift · {zname(zone)}")
    c1, c2 = st.columns([2, 1])
    cond = c2.selectbox("Demo conditions", order, index=0)
    w = by_label[cond]
    s0 = local_time(w)
    c1.markdown(f"#### Shift {s0:%H:%M}–{(s0 + pd.Timedelta(hours=6)):%H:%M}")
    c1.caption(f"Demo conditions: a real {cond.lower()} morning from the Ministry's data. In use, this page reads "
               "the live weather forecast and vehicle feeds and opens on your next shift automatically.")

    rows = shift_rows(w, zone).sort_values("risk_xgboost", ascending=False)
    high = rows[rows.risk_level == "High"]
    elev = rows[rows.risk_level == "Elevated"]

    # ---- 1. headline
    CAUSE = {"Weather forecast": "the weather forecast", "Segment history": "these stretches' crash history",
             "Road surface": "slippery road conditions", "Pavement": "pavement condition", "Time": "the time of day",
             "Traffic": "heavy traffic", "Location": "where these stretches are", "Road design": "road design",
             "Recent incidents": "recent incidents"}
    main_reason = CAUSE.get(high.reason_1_group.mode().iloc[0], "several factors") if len(high) else None
    if len(high):
        st.error(f"### High-risk shift: {len(high)} stretch{'es' if len(high) > 1 else ''} at High risk, "
                 f"mainly because of {main_reason}.")
    elif len(elev):
        st.warning(f"### Watch shift: no High-risk stretches, {len(elev)} to keep an eye on.")
    else:
        st.success("### Normal shift: no stretches at High or Elevated risk.")

    # ---- 2. the recommendation and the verdict
    st.subheader("Your move")
    scen = planned.get(w)
    own_move = incoming = None
    if scen is not None:
        m = mv[(mv.scenario == scen) & (mv.case == "All depots open")]
        own = m[m.from_zone == zone]
        inc = m[(m.to_zone == zone) & (m.from_zone != zone)]
        own_move = own.iloc[0] if len(own) else None
        incoming = inc.iloc[0] if len(inc) else None
        zrow = zp[(zp.scenario == scen) & (zp.case == "All depots open") & (zp.maintenance_zone == zone)].iloc[0]
        segs = patrol("dash_patrol_segments")
        covered = segs[(segs.scenario == scen) & (segs.case == "All depots open") &
                       (segs.maintenance_zone == zone) & (segs.response_min_saved > 0.5)].segment_id
        covered_rows = rows[rows.segment_id.isin(covered)]
    else:
        zrow, covered_rows = None, rows.head(1) if len(high) else rows.iloc[0:0]

    # target and verdict
    if own_move is not None:
        target_seg, target_zone = own_move.to_location, own_move.to_zone
        km = float(own_move.distance_km)
        if target_zone == zone:
            instruction = (f"Send 1 unit from **{depot(own_move.from_depot)}** to **{stretch(target_seg)}** ({target_seg}), "
                           f"{km:.0f} km away, about {km * MIN_PER_KM:.0f} min drive. Have it there when the shift starts.")
        else:
            instruction = (f"Lend 1 unit from **{depot(own_move.from_depot)}** to {zname(target_zone)}: "
                           f"wait at **{stretch(target_seg)}** ({target_seg}), {km:.0f} km away. "
                           f"The plan uses it to shorten response there.")
    elif len(high):
        top = high.iloc[0]
        target_seg, target_zone = top.segment_id, zone
        instruction = (f"Pre-position 1 unit near **{stretch(target_seg)}** ({target_seg}) before the shift starts. "
                       f"Use the nearest depot.")
    else:
        target_seg, target_zone = None, zone

    risky = covered_rows[covered_rows.risk_level.isin(["High", "Elevated"])]
    gain = float(zrow.response_min_saved) if zrow is not None else None
    if target_seg is None:
        verdict, why_v = "none", ""
    elif target_zone != zone:
        if len(high):
            verdict, why_v = "call", (f"This move helps {zname(target_zone)}, but your own zone has {len(high)} "
                                      f"High-risk stretch{'es' if len(high) > 1 else ''}. Agree it with that zone first.")
        else:
            verdict, why_v = "follow", f"Your zone is quiet this shift, so a unit can help {zname(target_zone)}."
    elif len(high) and not risky.risk_level.eq("High").any() and scen is not None:
        hs = segs[(segs.scenario == scen) & (segs.case == "All depots open") & segs.segment_id.isin(high.segment_id)]
        worst = float(hs.response_min_after.max()) if len(hs) else None
        if worst is not None and worst <= NEAR_MIN:
            verdict, why_v = "follow", (f"Your High-risk stretches are already within about {worst:.0f} min of a depot, "
                                        f"so this unit does more good here.")
        else:
            far = hs.sort_values("response_min_after", ascending=False).segment_id.head(2).map(stretch)
            verdict, why_v = "call", (f"This move does not cover your High-risk stretches ({', '.join(far)}). "
                                      "Consider placing the unit nearer to them.")
    elif (risky.risk_level.eq("High") & risky.confidence_band.eq("High")).any() and (gain is None or gain >= SMALL_GAIN_MIN):
        verdict, why_v = "follow", "It covers High-risk stretches and all three models agree."
    elif risky.risk_level.eq("High").any():
        verdict, why_v = "call", "It covers High-risk stretches, but the models do not fully agree. Check local conditions first."
    else:
        verdict, why_v = "optional", ("No High-risk stretches this shift. "
                                      + (f"The move still cuts response in your zone by about {gain:.0f} min, "
                                         "so follow it if a unit is free." if gain is not None and gain >= SMALL_GAIN_MIN
                                         else "Keeping units at the depot is fine."))

    with st.container(border=True):
        if target_seg is None:
            st.markdown("**No move needed.** Keep units at their depots and patrol as normal.")
        else:
            left, right = st.columns([3, 1])
            left.markdown(instruction)
            if target_zone == zone:
                tr = rows[rows.segment_id == target_seg]
                cover_txt = ", ".join(stretch(s) for s in risky.segment_id.head(3)) or stretch(target_seg)
                why = reasons(risky.iloc[0]) if len(risky) else (reasons(tr.iloc[0]) if len(tr) else "")
                left.markdown(f"**Why:** covers {cover_txt}. {why[:1].upper() + why[1:] if why else ''}")
            if zrow is not None and target_zone == zone:
                left.markdown(f"**What you gain:** help reaches likely serious incidents in your zone in about "
                              f"**{zrow.response_min_after:.0f} min instead of {zrow.response_min_before:.0f}**.")
            label = {"follow": "✅ Follow", "call": "⚠️ Your call", "optional": "➖ Optional"}[verdict]
            right.markdown(f"#### {label}")
            right.caption(why_v)

            st.markdown("**Quick check before you send it**")
            k = f"{w}_{zone}"
            a = st.checkbox("The unit is available (no breakdown, crew on shift)", key=f"a_{k}")
            b = st.checkbox("No road closure or detour on the way", key=f"b_{k}")
            c = st.checkbox("No emergency already needs this unit", key=f"c_{k}")
            b1, b2 = st.columns([1, 3])
            row = {"override_id": str(uuid.uuid4()), "logged_at_utc": datetime.now(timezone.utc),
                   "supervisor_id": supervisor.strip(), "window_start_utc": aware(w), "segment_id": target_seg,
                   "maintenance_zone": zone,
                   "model_risk_level": (risky.risk_level.iloc[0] if len(risky) else "Low"),
                   "model_risk": float(risky.risk_xgboost.iloc[0]) if len(risky) else 0.0,
                   "recommended_action": instruction.replace("**", ""), "note": ""}
            if b1.button("Confirm move", type="primary", disabled=not (a and b and c), key=f"ok_{k}"):
                if not supervisor.strip():
                    st.error("Enter your supervisor ID on the left first.")
                else:
                    log_decision({**row, "supervisor_decision": "Followed", "reason": "Checks passed"})
                    st.success("Move confirmed and logged.")
            if not (a and b and c):
                b2.caption("Tick all three to confirm. If any is not true, use “Can’t do it” below.")
            with st.expander("Can’t do it, or you know better"):
                st.caption("You never need approval to decline. Your reason is logged so the Ministry can learn from it.")
                with st.form(f"no_{k}"):
                    default_reason = 1 if not a else 2 if not b else 0 if not c else 3
                    reason = st.selectbox("Reason", ["Active emergency elsewhere", "Unit unavailable",
                                                     "Road closure, detour or work zone not in the data",
                                                     "Conditions differ from the forecast", "Other"], index=default_reason)
                    decision = st.radio("What you will do instead", ["Keep units at the depot",
                                        "Move a different unit or to a different place"], horizontal=True)
                    note = st.text_input("Note (optional)")
                    if st.form_submit_button("Log my decision"):
                        if not supervisor.strip():
                            st.error("Enter your supervisor ID on the left first.")
                        else:
                            log_decision({**row, "supervisor_decision": decision, "reason": reason, "note": note})
                            st.success("Logged. Your decision stands.")

    if incoming is not None:
        st.info(f"{zname(incoming.from_zone)} is lending you 1 unit from {depot(incoming.from_depot)}, "
                f"waiting at {stretch(incoming.to_location)} ({incoming.to_location}).")

    # ---- 3. watch list
    st.subheader("Stretches to watch this shift")
    def table(df):
        out = pd.DataFrame({
            "Where": df.segment_id.map(stretch), "Segment ID": df.segment_id, "Risk": df.risk_level,
            "Chance of a Critical incident": (df.risk_xgboost * 100).round(0).astype(int).astype(str) + "%",
            "Confidence": df.confidence_text, "Action": df.recommended_action, "Why": df.apply(reasons, axis=1)})
        st.dataframe(out, hide_index=True, use_container_width=True,
                     column_config={"Why": st.column_config.TextColumn(width="large")})
    if len(high):
        table(high.head(5))
        if len(high) > 5:
            with st.expander(f"Show all {len(high)} High-risk stretches"):
                table(high)
    else:
        st.caption("No High-risk stretches this shift.")
    if len(elev):
        with st.expander(f"Also keep an eye on: {len(elev)} Elevated stretch{'es' if len(elev) > 1 else ''}"):
            table(elev)

    hr_all, hr_hc = hit_rate("High"), hit_rate("High", "High")
    if hr_all:
        st.caption(f"How much to trust this: tested on 2023 data the model never saw, a High-risk stretch had a Critical incident during the shift "
                   f"{one_in(hr_all)} times ({one_in(hr_hc)} when all three models agreed), against about 1 in 100 "
                   f"for an ordinary stretch. Treat it as a strong heads-up, not a certainty.")

    # ---- 4. map
    segs = patrol("dash_patrol_segments").drop_duplicates("segment_id")
    mp = segs[segs.maintenance_zone == zone][["segment_id", "latitude", "longitude"]].merge(
        rows[["segment_id", "risk_level"]], on="segment_id", how="left")
    mp["color"] = mp.risk_level.map({"High": "#d9534f", "Elevated": "#f0ad4e"}).fillna("#9aa0a6")
    mp["size"] = mp.risk_level.map({"High": 2500, "Elevated": 1600}).fillna(700)
    if target_seg is not None and target_zone == zone:
        mp.loc[mp.segment_id == target_seg, ["color", "size"]] = ["#1f5fbf", 3500]
    with st.expander("Map of your zone (red = High, amber = Elevated, blue = where to send the unit)", expanded=True):
        st.map(mp, latitude="latitude", longitude="longitude", color="color", size="size")

# ---------------------------------------------------------------- page 2: decision log
elif page == "Decision log":
    st.header("Decision log")
    st.caption("Every move a supervisor confirmed or declined, with time and supervisor ID (US-007). "
               "Reviewed monthly to see where the recommendations do not fit conditions on the ground.")
    log = read_log()
    if log.empty:
        st.info("No decisions logged yet.")
    else:
        if zone:
            log = log[log.maintenance_zone == zone]
        st.dataframe(log, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", log.to_csv(index=False), "decision_log.csv")

# ---------------------------------------------------------------- page 3: what-if planning
else:
    st.header("What-if planning")
    st.caption("For planners and the Director: the three planning scenarios and the test of a Zone D depot closing (US-004).")
    zp, mv, asm = patrol("dash_patrol_zone_plan"), patrol("dash_patrol_moves"), patrol("dash_patrol_assumptions")
    c1, c2 = st.columns([2, 1])
    scen = c1.selectbox("Scenario", sorted(zp.scenario.unique()))
    cases = sorted(zp.case.unique(), key=lambda c: c != "All depots open")
    case = c2.radio("Depots", cases, horizontal=True,
                    format_func=lambda c: "All depots open" if c == "All depots open" else "Depot D-2 closed")
    z = zp[(zp.scenario == scen) & (zp.case == case)]
    m = mv[(mv.scenario == scen) & (mv.case == case)]
    st.caption(f"Sample conditions: the real morning of {shift_label(utc(z.shift_start_utc.iloc[0]))} (Eastern).")

    before = (z.expected_critical_windows * z.response_min_before).sum() / z.expected_critical_windows.sum()
    after = (z.expected_critical_windows * z.response_min_after).sum() / z.expected_critical_windows.sum()
    k1, k2, k3 = st.columns(3)
    k1.metric("Likely serious incidents this shift", f"about {z.expected_critical_windows.sum():.0f}")
    k2.metric("Average time for help to arrive", f"{after:.1f} min", f"{after - before:.1f} min vs no moves", delta_color="inverse")
    k3.metric("Units moved", f"{int(m.units.sum()) if len(m) else 0} of {int(z.units_based.sum())}")

    st.subheader("Moves (at most one per zone)")
    for _, r in m.iterrows():
        dest = "" if r.to_zone == r.from_zone else f" (lent to {zname(r.to_zone)})"
        st.markdown(f"- {zname(r.from_zone)}: 1 unit from {depot(r.from_depot)} to {stretch(r.to_location)} "
                    f"({r.to_location}), {r.distance_km:.0f} km{dest}")
    if m.empty:
        st.markdown("- No moves: keep all units at their depots.")

    st.subheader("By zone")
    show = pd.DataFrame({"Zone": z.maintenance_zone.map(zname),
                         "Likely serious incidents": z.expected_critical_windows.round(1),
                         "Units based": z.units_based, "Units waiting in zone": z.units_staged_in_zone,
                         "Help arrives, no moves (min)": z.response_min_before.round(1),
                         "Help arrives, with plan (min)": z.response_min_after.round(1),
                         "Minutes saved": z.response_min_saved.round(1)})
    st.dataframe(show, hide_index=True, use_container_width=True)

    sg = patrol("dash_patrol_segments")
    sg = sg[(sg.scenario == scen) & (sg.case == case)].copy()
    sg["size"] = 300 + 20000 * sg.risk
    sg["color"] = sg.response_min_saved.gt(0.5).map({True: "#1f5fbf", False: "#9aa0a6"})
    st.subheader("Map: bigger dot = higher risk; blue = help arrives faster with the plan")
    st.map(sg, latitude="latitude", longitude="longitude", size="size", color="color")
    with st.expander("Assumptions behind these numbers"):
        st.dataframe(asm, hide_index=True, use_container_width=True)
