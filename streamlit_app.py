import streamlit as st
import time
import pandas as pd
import openai
import pyotp
import plotly.graph_objects as go

from fcn_call_model import model, sysprompt, tools, Model, F_plot_sensitivity

st.set_page_config(
    page_title="GPT Function Calling",
    initial_sidebar_state="expanded",
)

if "TOTP_REQUIRED" not in st.secrets:
    st.error("Secrets not setup correctly. Must be done in the backend")
    st.stop()

if st.secrets.TOTP_REQUIRED == "NO":
    st.session_state["session_is_verified"] = True

if "session_is_verified" not in st.session_state:
    if "TOTP_SECRET" not in st.secrets:
        st.error("No TOTP secret found. Must be added in the backend")
        st.stop()
    st.session_state["session_is_verified"] = False


def wait_after_wrong_code_and_stop(text:str="Please wait", timeout_seconds:int=10):
    my_bar = st.progress(100, text=text)
    for percent_complete in range(99,0,-1):
        time.sleep(timeout_seconds/100)
        my_bar.progress(percent_complete + 1, text=text)
    my_bar.progress(0)
    st.stop()

if not st.session_state.session_is_verified:
    code = st.text_input(label="Enter the code", max_chars=6)
    if len(code) == 6:
        totp = pyotp.TOTP(s=st.secrets.TOTP_SECRET)
        if totp.verify(code):
            st.session_state["session_is_verified"] = True
            st.toast("Code OK", icon="🔑")
            time.sleep(1)
            st.rerun()
        else:
            st.error("code not valid")
            #wait_after_wrong_code_and_stop()
            #time.sleep(30)
            st.stop()            
    elif len(code) == 0:
        #st.error("enter the code")
        st.stop()    
    else:
        st.error("code not valid")
        #wait_after_wrong_code_and_stop()
        #time.sleep(30)
        st.stop()

if "timestamp_session_begin" not in st.session_state:
    st.session_state["timestamp_session_begin"] = time.time()
    st.toast(f"New Session started", icon="🎉")
else:
    st.toast(f"Session started {time.time()-st.session_state["timestamp_session_begin"]:.1f} seconds ago", icon="⏰")#⏳")


if "OPENAI_KEY" not in st.secrets:
    st.error("NO Key for OpenAI API found. Must be added in the backend")
    st.stop()

oai = openai.OpenAI(api_key=st.secrets.OPENAI_KEY)

prepared_prompts = [
    "What can you do?",
    "Summarize the model.",
    "How many outputs does the model have?",
]

if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = []

if "model" not in st.session_state:
    model.update_outputs()
    st.session_state["model"] = model

sensitivity_plots = []

with st.sidebar:
    btn_prompt = None
    with st.expander("Prompt Examples"):
        cols = st.columns(2)
        for i, prep_prompt in enumerate(prepared_prompts):
            col = cols[i % 2]
            if col.button(prep_prompt, use_container_width=True):
                btn_prompt = prep_prompt

    history_container = st.container()

    user_prompt = st.chat_input(placeholder="Ask the AI Assistant 🇺🇸🇧🇷🇯🇵🇩🇪🇰🇷🇪🇸🇮🇹🇫🇷🇨🇳...")
    if user_prompt or btn_prompt:
        prompt_to_gpt = user_prompt if user_prompt else btn_prompt
        with st.chat_message("User"):
            st.write(prompt_to_gpt)
        with st.status("Generating Response...", expanded=True) as status:
            st.markdown("### Creating Messages & Tools...")
            messages = [
                {"role": "system", "content": sysprompt(st.session_state["model"])},
                {"role": "user", "content": prompt_to_gpt},
            ]
            st.markdown("#### Messages:")
            st.write(messages)
            st.markdown("#### Tools:")
            st.write(tools)
            st.markdown("### Waiting for Response...")
            t1 = time.time()
            response = oai.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                tools=tools,
            )
            t_response = time.time() - t1
            st.write(f"Done. Took {t_response:.3f} seconds")
            st.markdown("### Response:")
            st.write(response.model_dump())
            status.update(
                label=f"Interaction complete after {t_response:.3} seconds!",
                state="complete",
                expanded=False,
            )

        with st.chat_message("ai"):
            if response.choices[0].message.content:
                # st.markdown("### Message:")
                st.markdown(response.choices[0].message.content)

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    # st.write(json.loads(tool_call.function.arguments), indent=2)
                    if tool_call.function.name == "update_model":
                        st.markdown(
                            f"### Calling function '{tool_call.function.name}' ✅"
                        )
                        st.session_state["model"] = Model.model_validate_json(
                            tool_call.function.arguments
                        )
                        st.session_state["model"].update_outputs()
                    elif tool_call.function.name == "optimize_model":
                        st.markdown(
                            f"### Calling function '{tool_call.function.name}' ✅"
                        )
                        opti_status = st.session_state["model"].optimize()
                        st.markdown(f"Optimization Result = {opti_status} (cost={st.session_state["model"].cost})")
                    elif tool_call.function.name == "plot_sensitivity":
                        st.markdown(
                            f"### Calling function '{tool_call.function.name}' ✅"
                        )
                        sensitivity_params = F_plot_sensitivity.model_validate_json(tool_call.function.arguments)
                        
                        
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=[1,2,3],
                                y=[6,3,9]
                            )
                        )
                        
                        sensitivity_plots.append(fig)
                        st.write(sensitivity_params)
                    else:
                        st.error(f"Unrecognized Function: {tool_call.function.name}")

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        cost_per_prompt_token = 0.150 / 1e6
        cost_per_completion_token = 0.600 / 1e6

        total_cost_USD = (prompt_tokens * cost_per_prompt_token) + (
            completion_tokens * cost_per_completion_token
        )

        st.session_state["prompt_history"].append(
            {
                "Prompt": prompt_to_gpt,
                "Prompt Tokens": prompt_tokens,
                "Completion Tokens": completion_tokens,
                "Cost USD": total_cost_USD,
            }
        )

        st.info(
            f"Token usage: {prompt_tokens}/{completion_tokens}/{response.usage.total_tokens} (prompt/completion/total)"
        )
        st.info(
            f"This interaction cost USD {total_cost_USD:.6f} using model '{response.model}'"
        )

    df_history = pd.DataFrame.from_records(st.session_state["prompt_history"])
    if "Cost USD" in df_history.columns:
        history_cost = df_history["Cost USD"].sum()
    else:
        history_cost = 0.0

    with history_container.expander(
        f"Prompt History: ({len(df_history)} entries), total cost = {float(history_cost):.6f}$ USD"
    ):
        if st.button("Clear", use_container_width=True, type="primary"):
            st.session_state["prompt_history"] = []
            df_history = pd.DataFrame()

        st.table(df_history)

m = st.session_state["model"]
st.markdown(f"### Model '{m.name}'")
st.markdown(f"##### Description: {m.descr}")


for i_plot, p in enumerate(sensitivity_plots):
    st.plotly_chart(p,key=f"sensitivity_chart_{i_plot}")

st.markdown(f"### Inputs")
st.dataframe(m.df_inputs, use_container_width=True, hide_index=True)
st.markdown(f"### Outputs")
st.dataframe(m.df_outputs, use_container_width=True, hide_index=True)
st.markdown(f"### Cost = {m.cost}")
st.write("Cost is the sum of distances between value and target for all outputs, if the respective target is active")

# with st.expander("Details"):
#     st.write(m.model_dump())
