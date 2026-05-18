import time
import json
import streamlit as st
import httpx
import pandas as pd

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Warehouse Refund Processing System", layout="wide")
st.title("Multi-Agent Warehouse Refund Processing System")

tab1, tab2, tab3, tab4 = st.tabs([
    "Process Refund", "Prompt Manager", "Analytics Dashboard", "Audit Logs"
])

# ─── TAB 1: Process Refund ───
with tab1:
    st.header("Submit Refund Request")

    col1, col2 = st.columns(2)
    with col1:
        customer_email = st.text_input("Customer Email", value="alice@example.com")
        order_id = st.number_input("Order ID", min_value=1, value=1, step=1)
    with col2:
        refund_reason = st.text_area("Refund Reason", value="Product not as described")

    if st.button("Submit Refund Request", type="primary"):
        with st.spinner("Processing warehouse refund request..."):
            try:
                response = httpx.post(f"{API_BASE}/refund", json={
                    "customer_email": customer_email,
                    "order_id": order_id,
                    "refund_reason": refund_reason,
                })
                task_data = response.json()
                task_id = task_data["task_id"]

                st.info(f"Task ID: {task_id}")

                # Poll for result
                max_polls = 30
                for i in range(max_polls):
                    time.sleep(2)
                    result_resp = httpx.get(f"{API_BASE}/refund/{task_id}")
                    result = result_resp.json()

                    if result.get("status") != "processing":
                        break

                if result.get("status") == "completed":
                    decision = result.get("decision", "unknown")
                    color_map = {
                        "approved": "green",
                        "denied": "red",
                        "partial": "orange",
                    }
                    badge_color = color_map.get(decision, "gray")
                    st.markdown(
                        f"### Decision: :{badge_color}[{decision.upper()}]"
                    )

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Refund Amount", f"${result.get('refund_amount', 0):.2f}")
                        st.write(f"**Customer:** {result.get('customer_name', 'N/A')}")
                        st.write(f"**Tier:** {result.get('customer_tier', 'N/A')}")
                    with col_b:
                        st.write(f"**Reason:** {result.get('policy_reason', 'N/A')}")
                        st.write(f"**Policy Applied:** {result.get('policy_applied', 'N/A')}")
                        st.write(f"**Validation:** {'Passed' if result.get('validation_passed') else 'Failed'}")

                elif result.get("status") == "error":
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("Request is still processing. Check back later.")
            except Exception as e:
                st.error(f"Connection error: {e}")

# ─── TAB 2: Prompt Manager ───
with tab2:
    st.header("Prompt Version Manager")

    prompt_names = ["supervisor", "validation_agent", "policy_agent", "communication_agent"]
    selected_prompt = st.selectbox("Select Prompt", prompt_names)

    if selected_prompt:
        try:
            resp = httpx.get(f"{API_BASE}/prompts/{selected_prompt}")
            data = resp.json()
            versions = data.get("versions", [])

            if versions:
                df = pd.DataFrame(versions)
                st.dataframe(df, use_container_width=True)

                # Activate version
                st.subheader("Activate Version")
                version_options = [v["version"] for v in versions]
                activate_version = st.selectbox("Version to activate", version_options)
                if st.button("Activate"):
                    resp = httpx.post(
                        f"{API_BASE}/prompts/{selected_prompt}/activate",
                        json={"version": activate_version},
                    )
                    st.success(f"Activated {selected_prompt} version {activate_version}")

                # Show active content
                active = next((v for v in versions if v.get("is_active")), None)
                if active:
                    st.subheader("Active Prompt Content")
                    st.code(active.get("content", ""), language="text")
            else:
                st.info("No versions found for this prompt.")
        except Exception as e:
            st.error(f"Error loading prompts: {e}")

    # Create new version
    st.subheader("Create New Version")
    with st.form("create_version_form"):
        new_version = st.text_input("Version (e.g., v1.1)")
        new_content = st.text_area("Prompt Content", height=200)
        new_description = st.text_input("Description")
        new_created_by = st.text_input("Created By")
        submitted = st.form_submit_button("Create Version")

        if submitted and new_version and new_content:
            try:
                resp = httpx.post(
                    f"{API_BASE}/prompts/{selected_prompt}/version",
                    json={
                        "version": new_version,
                        "content": new_content,
                        "description": new_description,
                        "created_by": new_created_by,
                    },
                )
                st.success(f"Created version {new_version}")
            except Exception as e:
                st.error(f"Error: {e}")

# ─── TAB 3: Analytics Dashboard ───
with tab3:
    st.header("Customer Analytics")

    analytics_customer_id = st.number_input("Customer ID", min_value=1, value=1, step=1, key="analytics_id")

    if st.button("Fetch Analytics"):
        try:
            resp = httpx.get(f"{API_BASE}/analytics/customer/{analytics_customer_id}")
            data = resp.json()

            if "error" not in data:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Spent", f"${data.get('total_spent', 0):.2f}")
                with col2:
                    st.metric("Avg Order Value", f"${data.get('average_order_value', 0):.2f}")
                with col3:
                    st.metric("Refund Rate", f"{data.get('refund_rate', 0):.2%}")
                with col4:
                    st.metric("Risk Score", f"{data.get('risk_score', 0):.2f}")
            else:
                st.warning(data["error"])
        except Exception as e:
            st.error(f"Error: {e}")

# ─── TAB 4: Audit Logs ───
with tab4:
    st.header("Audit Logs")

    if st.button("Refresh Logs"):
        try:
            resp = httpx.get(f"{API_BASE}/audit-logs")
            logs = resp.json()

            if logs:
                df = pd.DataFrame(logs)
                display_cols = ["timestamp", "agent_name", "tool_called", "status", "duration_ms"]
                available_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[available_cols], use_container_width=True)

                # Expandable details
                for i, log in enumerate(logs):
                    with st.expander(f"Log {log.get('id', i)} - {log.get('tool_called', 'N/A')}"):
                        st.json({"input_data": log.get("input_data"), "output_data": log.get("output_data")})
            else:
                st.info("No audit logs found.")
        except Exception as e:
            st.error(f"Error: {e}")
