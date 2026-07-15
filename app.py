import os
import json
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

# Load environmental variables
load_dotenv()

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.warning("⚠️ GROQ_API_KEY not found in environment variables. Please provide it in the sidebar.")
    return Groq(api_key=api_key)


# --- Core Groq Prompt Implementations (Using JSON Object Mode) ---

def analyze_kdp_metadata(client, title, subtitle, description):
    """Generates 7 long-tail keyword phrases and exactly 3 valid Amazon Store Category paths."""
    
    system_prompt = (
        "You are an expert Amazon KDP self-publishing specialist and SEO strategist.\n"
        "Analyze the provided book metadata to output search-optimized data.\n\n"
        
        "CRITICAL RULES FOR THE 7 BACKEND KEYWORDS:\n"
        "1. Generate exactly 7 backend search keyphrases.\n"
        "2. MUST be 'long-tail' phrases (e.g., 'enemies to lovers fantasy romance', NOT 'romance' or 'fantasy').\n"
        "3. Do NOT repeat words that are already in the Title or Subtitle. Amazon already indexes those words automatically.\n"
        "4. Focus on reader intent: character types, specific tropes, unique settings, tone/mood, and target audience.\n"
        "5. Each of the 7 phrases must be under 50 characters.\n\n"
        
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
        '  "optimization_reasoning": "A brief explanation of how these categories and keyword phrases target the Amazon algorithm."\n'
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


def suggest_seo_titles(client, draft_title, draft_subtitle, description):
    """Suggests three algorithm-optimized title and subtitle options."""
    system_prompt = (
        "You are an Amazon KDP Publishing & Algorithm expert. Your goal is to rewrite or optimize "
        "the user's draft title and subtitle to maximize click-through-rates (CTR) and organic discoverability.\n\n"
        "RULES:\n"
        "1. Keep the main title memorable, clear, and punchy.\n"
        "2. Subtitles must incorporate critical genre-identifying, trope-defining search keywords "
        "without looking like spam (e.g. 'A Psychological Suspense Thriller with an Unreliable Narrator').\n"
        "3. You must respond with a JSON object matching this exact schema:\n"
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
sidebar_key = st.sidebar.text_input("Enter Groq API Key:", type="password", help="If left empty, the app will try to read GROQ_API_KEY from environment files.")

# Resolve API Client
if sidebar_key:
    groq_client = Groq(api_key=sidebar_key)
else:
    groq_client = get_groq_client()

st.title("📚 Amazon KDP Metadata Optimizer")
st.write("Optimize your Amazon store presence by generating high-impact, long-tail backend tags, exact store categories, and click-optimized titles.")

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
    
    if not groq_client:
        st.info("Provide a Groq API key to begin optimization runs.")
        
    elif run_seo_analysis:
        if not title_input or not description_input:
            st.error("Please provide both a Title and a Description to perform SEO analysis.")
        else:
            with st.spinner("Analyzing metadata with Groq..."):
                try:
                    seo_results = analyze_kdp_metadata(groq_client, title_input, subtitle_input, description_input)
                    
                    st.success("Analysis Complete!")
                    
                    # Display 7 Long-tail Keywords
                    st.markdown("### 🏷️ 7 Long-Tail Backend Keyphrases")
                    st.info("Paste these into KDP's 7 keyword slots. They avoid repeating words in your title/subtitle and focus on real reader search patterns.")
                    
                    # Render keyphrases as a neat list with character counts
                    for i, tag in enumerate(seo_results["seven_backend_keywords"], 1):
                        char_len = len(tag)
                        color = "green" if char_len <= 50 else "red"
                        st.markdown(f"**Slot {i}:** `{tag}`  *(Length: <span style='color:{color};'>{char_len} chars</span>)*", unsafe_allow_html=True)
                    
                    st.write("") # Spacer
                    
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
                    
    elif run_title_generator:
        if not title_input or not description_input:
            st.error("Please provide both a Draft Title and a Description to generate variations.")
        else:
            with st.spinner("Generating algorithm-optimized titles..."):
                try:
                    title_results = suggest_seo_titles(groq_client, title_input, subtitle_input, description_input)
                    
                    st.success("Optimization Strategies Generated!")
                    st.markdown("### 💡 Recommended Hook Variations")
                    
                    for idx, option in enumerate(title_results["suggested_titles"], 1):
                        with st.expander(f"Option {idx}: {option['title']}", expanded=True):
                            st.markdown(f"**Suggested Title:** `{option['title']}`")
                            st.markdown(f"**Suggested Subtitle:** *{option['subtitle']}*")
                            st.markdown(f"**SEO Hook Strategy:** {option['strategy']}")
                            
                except Exception as e:
                    st.error(f"Error communicating with Groq: {e}")
    else:
        st.write("Enter your book metadata on the left and select an optimization pathway to begin.")