"""Receipt Sorter Pro: a private, local expense tracker.  Run: python -m streamlit run app_pro.py"""

import io
import json

import ollama
import pandas as pd
import streamlit as st

import receipt_core as core

st.set_page_config(page_title="Receipt Sorter Pro", page_icon="🧾", layout="wide")
core.init_db()


def installed_models():
    try:
        return [m.model for m in ollama.list().models]
    except Exception:
        return []


# ---------- Sidebar ----------
with st.sidebar:
    st.title("🧾 Receipt Sorter Pro")
    models = installed_models()
    if models:
        model = st.selectbox("Local model", models, index=models.index("qwen3:4b") if "qwen3:4b" in models else 0)
    else:
        model = st.text_input("Local model", "qwen3:4b")
        st.warning("Ollama not reachable. Start it first.")
    st.caption("🔒 Everything runs on this computer. No cloud, no API fees.")

tab_upload, tab_dash, tab_ledger, tab_rules, tab_export = st.tabs(
    ["📥 Upload", "📊 Dashboard", "✏️ Review & Edit", "⚙️ Vendor Rules", "📤 Export"]
)

# ---------- Upload ----------
with tab_upload:
    st.subheader("Add receipts")
    files = st.file_uploader("Drop .txt or text-based .pdf receipts", type=["txt", "pdf"], accept_multiple_files=True)
    pasted = st.text_area("...or paste a receipt", height=120)
    if st.button("Extract & save", type="primary"):
        items = [(f.name, f.getvalue()) for f in files or []]
        if pasted.strip():
            items.append(("pasted.txt", pasted.encode("utf-8")))
        if not items:
            st.info("Add at least one receipt.")
        bar = st.progress(0.0)
        for i, (name, data) in enumerate(items, 1):
            h = core.file_hash(data)
            if core.is_duplicate(h):
                st.warning(f"Skipped {name}: already saved (duplicate).")
            else:
                try:
                    text = core.read_file_text(name, data)
                    if not text.strip():
                        raise ValueError("no readable text (scanned PDF/image?)")
                    r = core.extract_receipt(text, model)
                    core.save_receipt(h, name, r)
                    flag = " ⚠️ please review" if r["needs_review"] else ""
                    st.success(f"{name}: {r['vendor']} | {r['date']} | {r['total']:.2f} | {r['category']}{flag}")
                except Exception as exc:
                    st.error(f"{name}: {exc}")
            bar.progress(i / len(items))

df = core.load_receipts()

# ---------- Dashboard ----------
with tab_dash:
    if df.empty:
        st.info("No receipts yet. Upload some first.")
    else:
        d = df.copy()
        d["dt"] = pd.to_datetime(d["date"], errors="coerce")
        d["currency"] = d["currency"].fillna("").astype(str).str.strip().str.upper()
        d["currency"] = d["currency"].replace("", "Unknown")
        c1, c2 = st.columns(2)
        cats = c1.multiselect("Categories", core.CATEGORIES, default=core.CATEGORIES)
        valid = d["dt"].dropna()
        if not valid.empty:
            rng = c2.date_input("Date range", (valid.min().date(), valid.max().date()))
            if len(rng) == 2:
                d = d[d["dt"].isna() | ((d["dt"].dt.date >= rng[0]) & (d["dt"].dt.date <= rng[1]))]
        d = d[d["category"].isin(cats)]

        m = st.columns(4)
        totals = core.currency_totals(d)
        total_display = " | ".join(
            f"{currency} {amount:,.2f}" for currency, amount in totals.items()
        ) or "0"
        average_by_currency = d.groupby("currency")["total"].mean().sort_index()
        average_display = " | ".join(
            f"{currency} {amount:,.2f}"
            for currency, amount in average_by_currency.items()
        ) or "0"
        m[0].metric("Total spent by currency", total_display)
        m[1].metric("Receipts", len(d))
        m[2].metric("Average by currency", average_display)
        m[3].metric("Needs review", int(d["needs_review"].sum()))

        left, right = st.columns(2)
        left.subheader("By category")
        category_totals = d.pivot_table(
            index="category",
            columns="currency",
            values="total",
            aggfunc="sum",
            fill_value=0,
        )
        left.bar_chart(category_totals)
        right.subheader("By month")
        d["month"] = d["dt"].dt.to_period("M").astype(str)
        monthly_totals = d.dropna(subset=["dt"]).pivot_table(
            index="month",
            columns="currency",
            values="total",
            aggfunc="sum",
            fill_value=0,
        )
        right.bar_chart(monthly_totals)
        st.subheader("Top vendors")
        vendors = d.groupby(["vendor", "currency"])["total"].agg(["count", "sum"])
        st.dataframe(
            vendors.sort_values("sum", ascending=False).head(10),
            use_container_width=True,
        )

# ---------- Review & Edit ----------
with tab_ledger:
    if df.empty:
        st.info("Nothing to review yet.")
    else:
        st.caption("Small models make mistakes. Fix any cell, tick 'delete' to remove a row, then save.")
        only_flagged = st.checkbox("Show only receipts flagged for review")
        view = df[df["needs_review"] == 1] if only_flagged else df
        view = view.drop(columns=["file_hash", "items", "added_at"]).assign(delete=False)
        view["needs_review"] = view["needs_review"].astype(bool)
        edited = st.data_editor(
            view, hide_index=True, use_container_width=True, disabled=["id", "filename"],
            column_config={"category": st.column_config.SelectboxColumn(options=core.CATEGORIES, required=True)},
        )
        if st.button("Save changes", type="primary"):
            core.update_receipts(edited)
            st.success("Saved.")
            st.rerun()
        pick = st.selectbox("View line items for", df["filename"] + " (#" + df["id"].astype(str) + ")")
        row = df[df["id"] == int(pick.split("#")[-1].rstrip(")"))].iloc[0]
        st.table(pd.DataFrame(json.loads(row["items"]) or [{"description": "-", "amount": 0}]))

# ---------- Rules ----------
with tab_rules:
    st.subheader("Vendor rules")
    st.write("If a vendor name contains the text, it is always put in your chosen category. This overrides the model for future uploads.")
    a, b, c = st.columns([3, 2, 1])
    pattern = a.text_input("Vendor contains", placeholder="e.g. uber")
    cat = b.selectbox("Category", core.CATEGORIES)
    if c.button("Add rule") and pattern.strip():
        core.add_rule(pattern, cat)
        st.rerun()
    for p, cg in core.get_rules().items():
        x, y = st.columns([5, 1])
        x.write(f"**{p}** → {cg}")
        if y.button("Delete", key=f"del_{p}"):
            core.delete_rule(p)
            st.rerun()

# ---------- Export ----------
with tab_export:
    if df.empty:
        st.info("Nothing to export yet.")
    else:
        out = df.drop(columns=["file_hash", "items"])
        st.download_button("Download CSV", out.to_csv(index=False).encode(), "receipts.csv", "text/csv")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xl:
            out.to_excel(xl, sheet_name="Receipts", index=False)
            export_data = out.copy()
            export_data["currency"] = (
                export_data["currency"].fillna("").astype(str).str.strip().str.upper()
            )
            export_data["currency"] = export_data["currency"].replace("", "Unknown")
            export_data.groupby(["currency", "category"])["total"].agg(
                ["count", "sum"]
            ).to_excel(xl, sheet_name="By category")
            export_data.assign(month=export_data["date"].str[:7]).groupby(
                ["currency", "month"]
            )["total"].sum().to_excel(xl, sheet_name="By month")
        st.download_button(
            "Download Excel report", buf.getvalue(), "receipt_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )