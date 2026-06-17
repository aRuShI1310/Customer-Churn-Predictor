import os

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

PERSONA_TONE = {
    "High-Spend VIP Buyers": (
        "exclusive early access to premium collections, VIP concierge support, "
        "and invitation-only previews — never generic discount codes"
    ),
    "High-Traffic System Consumers": (
        "bandwidth-optimized browsing, priority CDN routing, and a curated "
        "content feed tuned to heavy usage patterns"
    ),
    "Frequent Returners": (
        "streamlined returns, extended return windows, and loyalty credits "
        "that reward their repeat trust in the brand"
    ),
    "Dormant Casual Browsers": (
        "a gentle re-engagement nudge with personalized product picks and "
        "a low-friction welcome-back offer"
    ),
}

RETENTION_PROMPT = PromptTemplate(
    input_variables=[
        "cluster_persona",
        "age",
        "gender",
        "country",
        "membership",
        "churn_probability",
        "top_shap_reason",
        "persona_strategy",
    ],
    template="""You are a senior customer retention specialist at a global e-commerce platform.
Draft a hyper-targeted, highly empathetic retention email for a user at imminent churn risk.

Customer Profile:
- Persona Segment: {cluster_persona}
- Age: {age}
- Gender: {gender}
- Country: {country}
- Membership Tier: {membership}
- Churn Probability: {churn_probability}
- Primary Risk Driver: {top_shap_reason}

Persona-Specific Strategy: {persona_strategy}

Requirements:
1. Open with a warm, personal greeting that acknowledges their relationship with the brand.
2. Reference their persona segment naturally — do NOT use generic mass-marketing language.
3. Address the primary risk driver ({top_shap_reason}) with a concrete, empathetic solution.
4. Tailor the offer to their membership tier and spending profile.
5. Keep the tone sincere, respectful, and human — never pushy or robotic.
6. End with a clear, low-pressure call to action.
7. Write only the email body (subject line optional as first line).

Draft the retention email now:""",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("final_app_data.csv")
    df["ip"] = df["ip"].astype(str)
    return df


@st.cache_resource
def load_model():
    return joblib.load("xgb_churn_model.pkl")


def get_selected_user(df: pd.DataFrame, chosen_ip: str) -> pd.Series:
    return df[df["ip"] == chosen_ip].iloc[0]


def build_risk_bar_chart(selected_user: pd.Series) -> go.Figure:
    metrics = {
        "Total Spend ($)": selected_user["total_spend"],
        "Total Bytes": selected_user["total_bytes"],
        "Return Count": selected_user["return_count"],
        "Churn Risk (%)": selected_user["Churn_Probability"] * 100,
    }
    labels = list(metrics.keys())
    values = list(metrics.values())
    top_reason = selected_user["top_shap_reason"]
    colors = []
    for label in labels:
        if top_reason == "Total Spend" and "Spend" in label:
            colors.append("#EF553B")
        elif top_reason == "Total Bytes" and "Bytes" in label:
            colors.append("#EF553B")
        else:
            colors.append("#636EFA")

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:,.1f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Risk Parameters — Primary Driver: <b>{top_reason}</b>",
        yaxis_title="Value",
        height=420,
        margin=dict(t=60, b=40),
        showlegend=False,
    )
    return fig


def get_persona_strategy(persona: str) -> str:
    return PERSONA_TONE.get(
        persona,
        "a personalized retention offer aligned with their unique browsing and purchase behavior",
    )


def get_groq_api_key() -> str | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


def generate_retention_email(selected_user: pd.Series) -> str:
    persona = selected_user["cluster_persona"]
    user_data = {
        "cluster_persona": persona,
        "age": selected_user["age"],
        "gender": selected_user["gender"],
        "country": selected_user["country"],
        "membership": selected_user["membership"],
        "churn_probability": f"{selected_user['Churn_Probability'] * 100:.1f}%",
        "top_shap_reason": selected_user["top_shap_reason"],
        "persona_strategy": get_persona_strategy(persona),
    }
    llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=get_groq_api_key())
    chain = RETENTION_PROMPT | llm | StrOutputParser()
    return chain.invoke(user_data)


def deploy_campaign(ip: str, email_body: str, selected_user: pd.Series) -> None:
    record = {
        "ip": ip,
        "cluster_persona": selected_user["cluster_persona"],
        "Churn_Probability": selected_user["Churn_Probability"],
        "membership": selected_user["membership"],
        "email_body": email_body,
    }
    campaigns = pd.DataFrame([record])
    if os.path.exists("deployed_campaigns.csv"):
        existing = pd.read_csv("deployed_campaigns.csv")
        campaigns = pd.concat([existing, campaigns], ignore_index=True)
    campaigns.to_csv("deployed_campaigns.csv", index=False)


def render_kpi_cards(df: pd.DataFrame) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Logged Profiles", f"{df.shape[0]:,}")
    col2.metric("Average Customer Spend ($)", f"${df['total_spend'].mean():,.2f}")
    col3.metric("Average Returned Transactions", f"{df['return_count'].mean():.2f}")


def main() -> None:
    st.set_page_config(
        page_title="Churn Intelligence Platform",
        page_icon="📊",
        layout="wide",
    )

    st.title("Churn Intelligence Platform")
    st.caption("Segmentation · Predictive Risk · AI Retention Playbook")

    df = load_data()
    _ = load_model()

    high_risk_mask = df["Churn_Probability"] > 0.50
    high_risk_count = len(df[high_risk_mask])
    high_risk_df = (
        df[high_risk_mask]
        .sort_values("Churn_Probability", ascending=False)
        .reset_index(drop=True)
    )
    high_risk_ips = high_risk_df["ip"].tolist()

    tab1, tab2, tab3 = st.tabs(
        [
            "The Segmentation Dashboard",
            "The Predictive Risk Ledger",
            "AI Retention Playbook Window",
        ]
    )

    with tab1:
        st.header("The Segmentation Dashboard")
        render_kpi_cards(df)

        st.subheader("Customer Segment Landscape")
        fig = px.scatter(
            df,
            x="total_spend",
            y="total_bytes",
            color="cluster_persona",
            hover_data=["ip", "membership", "Churn_Probability"],
            labels={
                "total_spend": "Total Spend ($)",
                "total_bytes": "Total Bytes",
                "cluster_persona": "Persona Segment",
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(marker=dict(size=6, opacity=0.55))
        fig.update_layout(
            height=520,
            legend=dict(title="Cluster Persona", orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("The Predictive Risk Ledger")
        st.markdown(
            f"**{high_risk_count:,}** users flagged with churn probability above 50%, "
            "ranked from highest to lowest risk."
        )

        display_cols = [
            "ip",
            "cluster_persona",
            "Churn_Probability",
            "top_shap_reason",
            "total_spend",
            "total_bytes",
            "membership",
            "country",
        ]
        st.dataframe(
            high_risk_df[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Churn_Probability": st.column_config.NumberColumn(
                    "Churn Probability", format="%.4f"
                ),
                "total_spend": st.column_config.NumberColumn("Total Spend ($)", format="$%.2f"),
            },
        )

        st.divider()
        st.subheader("Individual Risk Parameter Analysis")

        chosen_ip = st.selectbox(
            "Select User IP Address",
            options=high_risk_ips,
            key="risk_ip_select",
        )

        selected_user = get_selected_user(df, chosen_ip)

        info_col, chart_col = st.columns([1, 2])
        with info_col:
            st.metric(
                "Churn Probability",
                f"{selected_user['Churn_Probability'] * 100:.2f}%",
            )
            st.write(f"**Persona:** {selected_user['cluster_persona']}")
            st.write(f"**Top SHAP Driver:** {selected_user['top_shap_reason']}")
            st.write(f"**Membership:** {selected_user['membership']}")
        with chart_col:
            st.plotly_chart(
                build_risk_bar_chart(selected_user),
                use_container_width=True,
                key=f"risk_chart_{chosen_ip}",
            )

    with tab3:
        st.header("AI Retention Playbook Window")
        st.markdown(
            "Select a high-risk customer to generate a persona-aware, empathetic retention email."
        )

        chosen_ip = st.selectbox(
            "Select High-Risk IP Address",
            options=high_risk_ips,
            key="playbook_ip_select",
        )

        selected_user = get_selected_user(df, chosen_ip)

        profile_cols = st.columns(4)
        profile_cols[0].metric(
            "Churn Risk",
            f"{selected_user['Churn_Probability'] * 100:.1f}%",
        )
        profile_cols[1].write(f"**Persona:** {selected_user['cluster_persona']}")
        profile_cols[2].write(f"**Membership:** {selected_user['membership']}")
        profile_cols[3].write(f"**Risk Driver:** {selected_user['top_shap_reason']}")

        with st.expander("Customer Context Variables", expanded=False):
            st.json(
                {
                    "cluster_persona": selected_user["cluster_persona"],
                    "age": str(selected_user["age"]),
                    "gender": selected_user["gender"],
                    "country": selected_user["country"],
                    "membership": selected_user["membership"],
                    "Churn_Probability": round(selected_user["Churn_Probability"], 4),
                    "top_shap_reason": selected_user["top_shap_reason"],
                }
            )

        draft_key = f"email_draft_{chosen_ip}"
        if st.session_state.get("active_playbook_ip") != chosen_ip:
            st.session_state["active_playbook_ip"] = chosen_ip

        if draft_key not in st.session_state:
            with st.spinner("Drafting retention email via LangChain PromptTemplate..."):
                st.session_state[draft_key] = generate_retention_email(selected_user)

        btn_col, _ = st.columns([1, 4])
        with btn_col:
            if st.button("Regenerate Draft", key=f"regen_draft_{chosen_ip}"):
                st.session_state[draft_key] = generate_retention_email(selected_user)
                st.rerun()

        action_col, deploy_col = st.columns([4, 1])
        with action_col:
            edited_email = st.text_area(
                "Retention Email Draft",
                value=st.session_state[draft_key],
                height=360,
                key=f"email_area_{chosen_ip}",
            )
        with deploy_col:
            st.write("")
            st.write("")
            if st.button("Deploy Campaign", type="primary", key=f"deploy_btn_{chosen_ip}"):
                deploy_campaign(chosen_ip, edited_email, selected_user)
                st.success(f"Campaign deployed for {chosen_ip}")
                st.balloons()


if __name__ == "__main__":
    main()
