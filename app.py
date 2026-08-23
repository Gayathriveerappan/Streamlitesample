import json
from datetime import datetime
from typing import Any

import streamlit as st


st.set_page_config(
    page_title="Topic Explainer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYSTEM_PROMPT = """You are an expert, patient topic explainer. Explain any topic the user asks about without assuming a fixed curriculum. Structure every response with:

1. A simple analogy that makes the core idea intuitive.
2. A clear technical explanation, introducing important terms.
3. A concise code example when code is relevant, with a short explanation of it.
4. A concrete real-world business use case.
5. A final section titled 'Why it matters' that connects the idea to practical decisions or outcomes.

Adapt the depth to the user's question. Be accurate, state uncertainty when needed, and use Markdown. Do not force a code example when the topic is not technical; say briefly why code is not relevant instead."""

PROVIDER_LABELS = {"OpenAI": "openai", "Claude": "claude"}


def init_state() -> None:
    defaults = {
        "provider": "OpenAI",
        "api_key": "",
        "models": [],
        "selected_model": None,
        "messages": [],
        "key_verified": False,
        "verification_message": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def discover_models(provider: str, api_key: str) -> list[str]:
    """List models available to this key; the request also verifies authentication."""
    if provider == "OpenAI":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        models = client.models.list().data
        model_ids = [model.id for model in models]
        preferred = [model_id for model_id in model_ids if any(
            marker in model_id.lower() for marker in ("gpt", "o1", "o3", "o4")
        )]
        return sorted(preferred or model_ids)

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    models = client.models.list(limit=100).data
    return sorted(model.id for model in models)


def explain_with_openai(api_key: str, model: str, messages: list[dict[str, str]]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
    )
    return response.choices[0].message.content or "The model returned an empty response."


def explain_with_claude(api_key: str, model: str, messages: list[dict[str, str]]) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return "\n".join(block.text for block in response.content if getattr(block, "type", "") == "text")


def request_explanation(provider: str, api_key: str, model: str, messages: list[dict[str, str]]) -> str:
    if provider == "OpenAI":
        return explain_with_openai(api_key, model, messages)
    return explain_with_claude(api_key, model, messages)


def format_transcript(messages: list[dict[str, Any]]) -> str:
    lines = ["Topic Explainer transcript", "=" * 28, ""]
    for message in messages:
        role = "You" if message["role"] == "user" else "Assistant"
        lines.extend([f"{role}:", message["content"], ""])
    return "\n".join(lines)


def clear_key() -> None:
    st.session_state.api_key = ""
    st.session_state.models = []
    st.session_state.selected_model = None
    st.session_state.key_verified = False
    st.session_state.verification_message = ""


def reset_provider_state() -> None:
    st.session_state.models = []
    st.session_state.selected_model = None
    st.session_state.key_verified = False
    st.session_state.verification_message = ""


def main() -> None:
    init_state()

    st.markdown(
        """
        <style>
        .stApp { background: #f5f7f2; }
        [data-testid="stSidebar"] { background: #18332f; }
        [data-testid="stSidebar"] * { color: #f4f0e6; }
        .brand { color: #d9a441; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12rem; text-transform: uppercase; }
        .hero { padding: 1.2rem 0 1.8rem; border-bottom: 1px solid #d7ddd4; margin-bottom: 1.4rem; }
        .hero h1 { color: #18332f; font-family: Georgia, serif; font-size: clamp(2.3rem, 5vw, 4.4rem); line-height: 1; margin: 0.25rem 0 0.65rem; }
        .hero p { color: #52645e; font-size: 1.05rem; max-width: 42rem; }
        .status { border-left: 3px solid #d9a441; padding: 0.4rem 0.8rem; color: #d9a441; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<div class="brand">Topic Explainer</div>', unsafe_allow_html=True)
        st.header("Connection")
        provider = st.radio(
            "Provider",
            list(PROVIDER_LABELS),
            key="provider",
            on_change=reset_provider_state,
        )
        api_key = st.text_input(
            f"{provider} API key",
            type="password",
            value=st.session_state.api_key,
            placeholder="Paste your key here",
            help="Your key is used only for requests from this session and is not written to disk.",
        )
        st.session_state.api_key = api_key.strip()

        verify_col, clear_col = st.columns(2)
        with verify_col:
            verify_clicked = st.button("Verify key", type="primary", use_container_width=True)
        with clear_col:
            st.button("Clear key", use_container_width=True, on_click=clear_key)

        if verify_clicked:
            if not st.session_state.api_key:
                st.session_state.verification_message = "Enter an API key first."
                st.session_state.key_verified = False
            else:
                with st.spinner("Checking access and discovering models..."):
                    try:
                        discovered = discover_models(provider, st.session_state.api_key)
                        st.session_state.models = discovered
                        st.session_state.selected_model = discovered[0] if discovered else None
                        st.session_state.key_verified = bool(discovered)
                        st.session_state.verification_message = (
                            f"Verified. {len(discovered)} model(s) available."
                            if discovered else "Key worked, but no models were returned."
                        )
                    except Exception as error:
                        st.session_state.models = []
                        st.session_state.selected_model = None
                        st.session_state.key_verified = False
                        st.session_state.verification_message = f"Could not verify key: {error}"

        if st.session_state.verification_message:
            if st.session_state.key_verified:
                st.success(st.session_state.verification_message)
            else:
                st.error(st.session_state.verification_message)

        if st.session_state.models:
            model_index = st.session_state.models.index(st.session_state.selected_model) if st.session_state.selected_model in st.session_state.models else 0
            st.session_state.selected_model = st.selectbox(
                "Model", st.session_state.models, index=model_index, key="model_picker"
            )
        else:
            st.caption("Verify a key to discover and select available models.")

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption("Keys stay in session memory only. Provider usage costs may apply.")

    st.markdown(
        '<div class="hero"><div class="brand">Ask anything</div><h1>Make difficult ideas click.</h1><p>One conversation for intuitive analogies, rigorous detail, useful code, and the business reason behind it all.</p></div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.key_verified:
        st.info("Connect a provider in the sidebar to start exploring.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.download_button(
                    "Download explanation",
                    data=message["content"],
                    file_name=f"explanation-{datetime.now():%Y%m%d-%H%M%S}.md",
                    mime="text/markdown",
                    key=f"download-{message['id']}",
                )

    prompt = st.chat_input("What would you like explained?")
    if prompt:
        if not st.session_state.key_verified or not st.session_state.selected_model:
            st.warning("Verify an API key and choose an available model first.")
            return

        user_message = {"role": "user", "content": prompt, "id": datetime.now().timestamp()}
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    api_messages = [
                        {"role": item["role"], "content": item["content"]}
                        for item in st.session_state.messages
                    ]
                    answer = request_explanation(
                        provider, st.session_state.api_key, st.session_state.selected_model, api_messages
                    )
                    st.markdown(answer)
                    answer_id = datetime.now().timestamp()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "id": answer_id,
                    })
                    st.download_button(
                        "Download explanation",
                        data=answer,
                        file_name=f"explanation-{datetime.now():%Y%m%d-%H%M%S}.md",
                        mime="text/markdown",
                        key=f"download-{answer_id}",
                    )
                except Exception as error:
                    st.error(f"The explanation could not be generated: {error}")

    if st.session_state.messages:
        st.download_button(
            "Download full transcript",
            data=format_transcript(st.session_state.messages),
            file_name=f"topic-explainer-{datetime.now():%Y%m%d-%H%M%S}.md",
            mime="text/markdown",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
