from pathlib import Path
import numpy as np
import pandas as pd

def load_csvs(data_dir="."):
    data_dir = Path(data_dir)
    opp = pd.read_csv(data_dir / "opportunity.csv")
    acct = pd.read_csv(data_dir / "account.csv")
    lead = pd.read_csv(data_dir / "lead.csv")
    contact = pd.read_csv(data_dir / "contact.csv")
    orders = pd.read_csv(data_dir / "order.csv")
    users = pd.read_csv(data_dir / "user.csv")

    opp["created_date"] = pd.to_datetime(opp["created_date"], errors="coerce")
    opp["close_date"] = pd.to_datetime(opp["close_date"], errors="coerce")
    contact["created_date"] = pd.to_datetime(contact["created_date"], errors="coerce")
    orders["created_date"] = pd.to_datetime(orders["created_date"], errors="coerce")
    users["hire_date"] = pd.to_datetime(users["hire_date"], errors="coerce")
    lead["created_date"] = pd.to_datetime(lead["created_date"], errors="coerce")

    return opp, acct, lead, contact, orders, users


def build_features(data_dir="."):
    opp, acct, lead, contact, orders, users = load_csvs(data_dir)

    # Only closed opportunities are used because their final outcome is known.
    # Closed Won = 1, Closed Lost = 0.
    closed = opp[opp["stage_name"].isin(["Closed Won", "Closed Lost"])].copy()
    closed["target"] = (closed["stage_name"] == "Closed Won").astype(int)
    closed = closed.sort_values("created_date").reset_index(drop=True)

    account_cols = [
        "id", "type", "industry", "annual_revenue", "employee_count",
        "billing_state", "billing_country", "is_active"
    ]
    a = acct[account_cols].copy()
    a.columns = [
        "account_id", "account_type", "industry", "annual_revenue",
        "employee_count", "billing_state", "billing_country",
        "account_active"
    ]
    df = closed.merge(a, on="account_id", how="left")

    # Date and numeric business features.
    df["created_year"] = df["created_date"].dt.year
    df["created_month"] = df["created_date"].dt.month
    df["created_quarter"] = df["created_date"].dt.quarter
    df["created_dow"] = df["created_date"].dt.dayofweek

    df["amount_log"] = np.log1p(df["amount"].clip(lower=0))
    df["revenue_log"] = np.log1p(df["annual_revenue"].clip(lower=0))
    df["emp_log"] = np.log1p(df["employee_count"].clip(lower=0))
    df["amount_revenue_ratio"] = df["amount"] / (df["annual_revenue"] + 1)
    df["amount_per_employee"] = df["amount"] / (df["employee_count"] + 1)

    # Owner profile. Raw owner_id is deliberately not used as a model feature.
    u = users[["id", "role", "department", "is_active", "hire_date"]].copy()
    u.columns = ["owner_id", "owner_role", "owner_department",
                 "owner_active", "hire_date"]
    df = df.merge(u, on="owner_id", how="left")
    df["owner_tenure_days"] = (
        df["created_date"] - df["hire_date"]
    ).dt.days.clip(lower=0)

    # Explicitly convert boolean CRM fields to numeric 0/1.
    # This prevents sklearn's numeric imputer from receiving mixed/object data.
    for bool_col in ["account_active", "owner_active"]:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].fillna(False).astype(int)

    # Lead information. Raw lead status is excluded because it can reflect
    # downstream conversion activity.
    l = lead[
        ["converted_opportunity_id", "lead_source", "created_date"]
    ].copy()
    l = l.rename(columns={
        "converted_opportunity_id": "id",
        "lead_source": "lead_source_from_lead",
        "created_date": "lead_created_date"
    })
    df = df.merge(l, on="id", how="left")
    df["lead_age_days"] = (
        df["created_date"] - df["lead_created_date"]
    ).dt.days.clip(lower=0)

    # Contacts known before the opportunity was created.
    c = contact[["account_id", "created_date", "is_primary"]].copy()
    c = c.sort_values(["created_date"])
    c["contact_count_cum"] = c.groupby("account_id").cumcount() + 1
    c["primary_count_cum"] = c.groupby("account_id")["is_primary"].cumsum()

    base = (
        df[["account_id", "created_date"]]
        .reset_index()
        .sort_values("created_date")
    )
    c2 = c[
        ["account_id", "created_date",
         "contact_count_cum", "primary_count_cum"]
    ].sort_values("created_date")

    matched = pd.merge_asof(
        base,
        c2,
        on="created_date",
        by="account_id",
        direction="backward",
        allow_exact_matches=False
    ).sort_values("index")

    df["prior_contact_count"] = matched["contact_count_cum"].fillna(0).to_numpy()
    df["prior_primary_contact_count"] = (
        matched["primary_count_cum"].fillna(0).to_numpy()
    )

    # Prior customer orders known before opportunity creation.
    o = orders[["account_id", "created_date", "total_amount"]].copy()
    o = o.sort_values(["created_date"])
    o["order_count_cum"] = o.groupby("account_id").cumcount() + 1
    o["order_value_cum"] = o.groupby("account_id")["total_amount"].cumsum()

    base = (
        df[["account_id", "created_date"]]
        .reset_index()
        .sort_values("created_date")
    )
    o2 = o[
        ["account_id", "created_date", "order_count_cum", "order_value_cum"]
    ].sort_values("created_date")

    matched = pd.merge_asof(
        base,
        o2,
        on="created_date",
        by="account_id",
        direction="backward",
        allow_exact_matches=False
    ).sort_values("index")

    df["prior_order_count"] = matched["order_count_cum"].fillna(0).to_numpy()
    df["prior_order_value"] = matched["order_value_cum"].fillna(0).to_numpy()

    # Historical win rates are calculated only from opportunities created
    # earlier than the current opportunity. This avoids target leakage.
    global_rate = df["target"].mean()

    df["prior_account_opps"] = df.groupby("account_id").cumcount()
    prior_account_wins = (
        df.groupby("account_id")["target"].cumsum() - df["target"]
    )
    df["prior_account_win_rate"] = (
        prior_account_wins + global_rate * 5
    ) / (df["prior_account_opps"] + 5)

    df["prior_owner_opps"] = df.groupby("owner_id").cumcount()
    prior_owner_wins = (
        df.groupby("owner_id")["target"].cumsum() - df["target"]
    )
    df["prior_owner_win_rate"] = (
        prior_owner_wins + global_rate * 10
    ) / (df["prior_owner_opps"] + 10)

    df["prior_source_opps"] = df.groupby("lead_source").cumcount()
    prior_source_wins = (
        df.groupby("lead_source")["target"].cumsum() - df["target"]
    )
    df["prior_source_win_rate"] = (
        prior_source_wins + global_rate * 10
    ) / (df["prior_source_opps"] + 10)

    # IMPORTANT:
    # stage_name, probability, close_date, days_to_close and raw account_id
    # are not model inputs. Stage/probability directly reveal the outcome for
    # closed records, while close-date-derived fields are not available when
    # making a genuine forecast.

    feature_cols = [
        "amount", "amount_log", "lead_source", "owner_role",
        "owner_department", "account_type", "industry",
        "created_year", "created_month", "created_quarter", "created_dow",
        "annual_revenue", "revenue_log", "employee_count", "emp_log",
        "account_active", "billing_state", "billing_country",
        "owner_active", "owner_tenure_days", "lead_source_from_lead",
        "lead_age_days", "prior_contact_count",
        "prior_primary_contact_count", "prior_order_count",
        "prior_order_value", "prior_account_opps",
        "prior_account_win_rate", "prior_owner_opps",
        "prior_owner_win_rate", "prior_source_opps",
        "prior_source_win_rate", "amount_revenue_ratio",
        "amount_per_employee"
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]
    return df, feature_cols
