import os
import json
import requests
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load environmental variables (reads .env locally)
load_dotenv()

# --- Helper Functions: Clients ---

def get_groq_client(sidebar_key=None):
    """
    Safely retrieves or initializes the Groq client.
    Returns None if no key is supplied, preventing application startup crashes.
    """
    if sidebar_key:
        return Groq(api_key=sidebar_key)

    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return Groq(api_key=api_key)

    return None


def get_tavily_key(sidebar_key=None):
    """
    Safely retrieves the Tavily API key from the sidebar input or environment.
    Returns None if unavailable (trend search is optional, so this must not crash the app).
    """
    if sidebar_key:
        return sidebar_key
    return os.getenv("TAVILY_API_KEY")


# --- Helper Functions: Tavily Web Search (Trend Grounding) ---

def search_trending_kdp_insights(tavily_key, genre_query):
    """
    Queries Tavily's search API for current Amazon KDP algorithm behavior,
    trending tropes, and trending backend keywords relevant to the given genre/niche.
    Returns a dict with a synthesized answer + supporting snippets, or an error dict.
    """
    if not tavily_key or not genre_query:
        return None

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": tavily_key,
        "query": f"Amazon KDP {genre_query} trending tropes keywords categories algorithm 2026",
        "search_depth": "advanced",
        "max_results": 6,
        "include_answer": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json()

        snippets = []
        for r in data.get("results", []):
            title = r.get("title", "Untitled")
            content = (r.get("content") or "")[:350]
            source_url = r.get("url", "")
            snippets.append({"title": title, "content": content, "url": source_url})

        return {
            "answer": data.get("answer", ""),
            "snippets": snippets,
        }
    except Exception as e:
        return {"error": str(e)}


def format_trend_context_for_prompt(trend_data):
    """Formats Tavily results into a compact text block to inject into the LLM prompt."""
    if not trend_data or trend_data.get("error"):
        return ""

    lines = []
    if trend_data.get("answer"):
        lines.append(f"Summary: {trend_data['answer']}")
    for s in trend_data.get("snippets", []):
        lines.append(f"- {s['title']}: {s['content']}")

    return "\n".join(lines)


# --- Helper Functions: Groq / LLM ---

def analyze_kdp_metadata(client, title, subtitle, description, trend_context=None):
    """Generates 7 long-tail keyword phrases and exactly 3 valid Amazon Store Category paths,
    optionally grounded in live web trend research."""

    trend_block = ""
    if trend_context:
        trend_block = (
            "\n\nLIVE WEB TREND RESEARCH (via Tavily search, current as of today):\n"
            f"{trend_context}\n\n"
            "Use this real-time research to prioritize backend keywords and categories that "
            "reflect what readers are ACTUALLY searching for and what's trending right now. "
            "Don't just describe the trends back — apply them to the specific book below.\n"
        )

    system_prompt = (
        "You are an expert Amazon KDP self-publishing specialist and SEO strategist.\n"
        "Analyze the provided book metadata to output search-optimized data.\n"
        f"{trend_block}\n"
        "CRITICAL RULES FOR THE 7 BACKEND KEYWORDS:\n"
        "1. Generate exactly 7 backend search keyphrases.\n"
        "2. MUST be 'long-tail' phrases (e.g., 'enemies to lovers fantasy romance', NOT 'romance' or 'fantasy').\n"
        "3. Do NOT repeat words that are already in the Title or Subtitle. Amazon already indexes those words automatically.\n"
        "4. Focus on reader intent: character types, specific tropes, unique settings, tone/mood, and target audience.\n"
        "5. Each of the 7 phrases must be under 50 characters.\n"
        "6. If live trend research is provided above, weave in currently trending tropes/terms where relevant.\n\n"

        "CRITICAL RULES FOR CATEGORIES:\n"
        "1. Provide exactly 3 valid, standard Amazon Store browse categories (e.g. 'Kindle Store > Kindle eBooks > Literature & Fiction > Genre Fiction > Coming of Age').\n"
        "2. Ensure the category paths are highly specific, drilling down to the deepest possible sub-category to target low-competition niches.\n\n"

        "You must respond with a JSON object matching this exact schema:\n"
        "{\n"
        '  "suggested_categories": [\n'
        '     "Kindle Store > Kindle eBooks > [Category] > [Subcategory] > [Specific Niche]",\n'
        '     "Kindle Store > Kindle eBooks > [Category] > [Subcategory] > [Specific Niche]",\n'
        '     "Kindle Store > Kindle eBooks > [Category] > [Subcategory] > [Specific Niche]"\n'
        '  ],\n'
        '  "seven_backend_keywords": [\n'
        '     "long tail phrase 1 under 50 chars",\n'
        '     "long tail phrase 2 under 50 chars",\n'
        '     ...\n'
        '  ],\n'
        '  "optimization_reasoning": "A brief explanation of how these categories and keyword phrases target the Amazon algorithm, referencing live trend data if used."\n'
        "}"
    )

    user_prompt = f"""
    Title: {title}
    Subtitle: {subtitle or 'N/A'}
    Description/Excerpt: {description}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.2
    )

    return json.loads(response.choices[0].message.content)


def suggest_seo_titles(client, draft_title, draft_subtitle, description, trend_context=None):
    """Suggests three algorithm-optimized title and subtitle options, optionally grounded in live web trends."""

    trend_block = ""
    if trend_context:
        trend_block = (
            "\n\nLIVE WEB TREND RESEARCH (via Tavily search, current as of today):\n"
            f"{trend_context}\n\n"
            "Use this to inform which genre-signaling words and phrasing styles are currently "
            "resonating with readers and performing well in titles/subtitles.\n"
        )

    system_prompt = (
        "You are an Amazon KDP Publishing & Algorithm expert. Your goal is to rewrite or optimize "
        "the user's draft title and subtitle to maximize click-through-rates (CTR) and organic discoverability.\n"
        f"{trend_block}\n"
        "RULES:\n"
        "1. Keep the main title memorable, clear, and punchy.\n"
        "2. Subtitles must incorporate critical genre-identifying, trope-defining search keywords "
        "without looking like spam (e.g. 'A Psychological Suspense Thriller with an Unreliable Narrator').\n"
        "3. If live trend research is provided above, factor in currently trending phrasing/tropes where genuinely fitting.\n"
        "4. You must respond with a JSON object matching this exact schema:\n"
        "{\n"
        '  "suggested_titles": [\n'
        '    {\n'
        '      "title": "Optimized Title Here",\n'
        '      "subtitle": "Search-optimized subtitle here",\n'
        '      "strategy": "Why this specific combination works for the algorithm."\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    user_prompt = f"""
    Current Draft Title: {draft_title}
    Current Draft Subtitle: {draft_subtitle or 'None provided'}
    Book Description: {description}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.7
    )

    return json.loads(response.choices[0].message.content)


# --- Streamlit Frontend UI ---

st.set_page_config(page_title="Amazon KDP MetaOptimizer", page_icon="📚", layout="wide")

# Sidebar Configuration
st.sidebar.title("🔑 Configuration")
sidebar_key = st.sidebar.text_input(
    "Enter Groq API Key:",
    type="password",
    help="If you haven't added GROQ_API_KEY to your Streamlit secrets, enter it here."
)

st.sidebar.write("---")
st.sidebar.subheader("📈 Live Trend Grounding (Tavily)")
tavily_sidebar_key = st.sidebar.text_input(
    "Enter Tavily API Key:",
    type="password",
    help="Optional. If you haven't added TAVILY_API_KEY to your Streamlit secrets, enter it here. Get a free key at tavily.com."
)
use_trend_search = st.sidebar.checkbox(
    "Ground suggestions in live web trends",
    value=True,
    help="When enabled, the app searches the web via Tavily for current KDP algorithm behavior and trending tropes/keywords before generating suggestions."
)
genre_query_input = st.sidebar.text_input(
    "Genre / Niche for trend search",
    placeholder="e.g., cozy mystery, dark romantasy, cyberpunk thriller",
    help="Used to search the web for what's trending in this specific niche right now."
)

# Safely resolve API Clients (will remain None on startup if no key is supplied)
groq_client = get_groq_client(sidebar_key)
tavily_key = get_tavily_key(tavily_sidebar_key)

st.title("📚 Amazon KDP Metadata Optimizer")
st.write("Optimize your Amazon store presence by generating high-impact, long-tail backend tags, exact store categories, and click-optimized titles — now grounded in live web trend research.")

# Layout with two main columns
col_input, col_output = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📝 Draft Metadata")

    title_input = st.text_input("Book Title", placeholder="e.g., The Silent Whispers", value="The Silent Whispers")
    subtitle_input = st.text_input("Book Subtitle (Optional)", placeholder="e.g., A Thrilling Mystery Novel", value="")

    description_input = st.text_area(
        "Book Excerpt / Description",
        placeholder="Paste your blurb, plot summary, or back cover copy here...",
        height=250,
        value="A reclusive detective is forced to confront her past when a series of cold cases are reopened by an anonymous copycat killer. To save the next victim, she must solve the riddles hidden within her own family's history."
    )

    st.write("---")

    # Action buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_seo_analysis = st.button("🚀 Generate Tags & Categories", use_container_width=True, type="primary")
    with col_btn2:
        run_title_generator = st.button("💡 Optimize Title & Subtitle", use_container_width=True)

with col_output:
    st.subheader("🎯 Optimization Output")

    # Lazy execution checks
    if run_seo_analysis or run_title_generator:
        if not groq_client:
            st.error("🔑 **Groq API Key Missing!** Please enter your Groq API Key in the sidebar or configure it in your Streamlit Secrets to proceed.")
        elif not title_input or not description_input:
            st.error("⚠️ Please provide both a Title and a Description to proceed.")
        else:
            # Step 0: Optional live trend search via Tavily
            trend_data = None
            trend_context = None
            if use_trend_search:
                if not tavily_key:
                    st.warning("📈 Trend grounding is enabled but no Tavily API key was found — skipping live trend search and using the model's base knowledge instead.")
                else:
                    effective_genre_query = genre_query_input or title_input
                    with st.spinner(f"Searching the web for trends in '{effective_genre_query}'..."):
                        trend_data = search_trending_kdp_insights(tavily_key, effective_genre_query)
                        if trend_data and trend_data.get("error"):
                            st.warning(f"⚠️ Tavily search failed ({trend_data['error']}) — continuing without live trend data.")
                            trend_data = None
                        elif trend_data:
                            trend_context = format_trend_context_for_prompt(trend_data)

            # Display trend panel if we got results
            if trend_data and not trend_data.get("error"):
                with st.expander("📈 Live Market Trends (via Tavily)", expanded=False):
                    if trend_data.get("answer"):
                        st.markdown(f"**Summary:** {trend_data['answer']}")
                    for s in trend_data.get("snippets", []):
                        st.markdown(f"- **{s['title']}** — {s['content']}")
                        if s.get("url"):
                            st.caption(s["url"])

            # Route 1: KDP Tag & Category Generation
            if run_seo_analysis:
                with st.spinner("Analyzing metadata with Groq..."):
                    try:
                        seo_results = analyze_kdp_metadata(
                            groq_client, title_input, subtitle_input, description_input,
                            trend_context=trend_context
                        )

                        st.success("Analysis Complete!" + (" (grounded in live trends)" if trend_context else ""))

                        # Display 7 Long-tail Keywords
                        st.markdown("### 🏷️ 7 Long-Tail Backend Keyphrases")
                        st.info("Paste these into KDP's 7 keyword slots. They avoid repeating words in your title/subtitle and focus on real reader search patterns.")

                        # Render keyphrases as a neat list with character counts
                        for i, tag in enumerate(seo_results["seven_backend_keywords"], 1):
                            char_len = len(tag)
                            color = "green" if char_len <= 50 else "red"
                            st.markdown(f"**Slot {i}:** `{tag}`  *(Length: <span style='color:{color};'>{char_len} chars</span>)*", unsafe_allow_html=True)

                        st.write("")  # Spacer

                        # Display 3 Categories
                        st.markdown("### 📁 Standard Amazon Store Categories")
                        st.warning("You can choose exactly 3 categories on your KDP dashboard. Use these paths to find them:")
                        for cat in seo_results["suggested_categories"]:
                            st.markdown(f"* **`{cat}`**")

                        # Display Strategy
                        st.markdown("### 🧠 Strategic Reasoning")
                        st.write(seo_results["optimization_reasoning"])

                    except Exception as e:
                        st.error(f"Error communicating with Groq: {e}")

            # Route 2: Title and Subtitle Optimization
            elif run_title_generator:
                with st.spinner("Generating algorithm-optimized titles..."):
                    try:
                        title_results = suggest_seo_titles(
                            groq_client, title_input, subtitle_input, description_input,
                            trend_context=trend_context
                        )

                        st.success("Optimization Strategies Generated!" + (" (grounded in live trends)" if trend_context else ""))
                        st.markdown("### 💡 Recommended Hook Variations")

                        for idx, option in enumerate(title_results["suggested_titles"], 1):
                            with st.expander(f"Option {idx}: {option['title']}", expanded=True):
                                st.markdown(f"**Suggested Title:** `{option['title']}`")
                                st.markdown(f"**Suggested Subtitle:** *{option['subtitle']}*")
                                st.markdown(f"**SEO Hook Strategy:** {option['strategy']}")

                    except Exception as e:
                        st.error(f"Error communicating with Groq: {e}")
    else:
        # Prompt when the application is idle
        if not groq_client:
            st.info("👈 Enter your Groq API Key in the sidebar to activate the optimizer.")
        else:
            st.write("Enter your book metadata on the left and select an optimization pathway to begin.")
