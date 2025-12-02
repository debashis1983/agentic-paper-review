"""
🎓 Agentic Paper Reviewer - Streamlit UI
=========================================

A beautiful, interactive UI for the LangGraph-based paper review system.
Features real-time streaming of agent progress and professional styling.

Usage:
    streamlit run streamlit_app.py

Requirements:
    pip install streamlit streamlit-extras watchdog
"""

import os
import sys
import json
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
import time

import streamlit as st

# Try to import streamlit-extras, but make it optional
try:
    from streamlit_extras.add_vertical_space import add_vertical_space
except ImportError:
    def add_vertical_space(n):
        for _ in range(n):
            st.write("")

# Import the agent
from agent import create_paper_reviewer_graph, ReviewerState, PaperMetadata


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Agentic Paper Reviewer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Dark theme styling */
    .main { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
    
    /* Score display */
    .score-display {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d9ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    
    /* Review sections */
    .review-section {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 20px 25px;
        margin: 15px 0;
        border-left: 4px solid #667eea;
    }
    .review-section.summary { border-left-color: #00d9ff; }
    .review-section.strengths { border-left-color: #00ff88; }
    .review-section.weaknesses { border-left-color: #ffd93d; }
    .review-section.detailed { border-left-color: #667eea; }
    .review-section.questions { border-left-color: #ff8c42; }
    .review-section.references { border-left-color: #764ba2; }
    .review-section.minor { border-left-color: #8892b0; }
    .review-section.recommendation { border-left-color: #e94560; }
    
    .review-section h3 {
        color: #00d9ff;
        margin: 0 0 15px 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Progress steps */
    .step-complete { color: #00ff88; }
    .step-active { color: #00d9ff; animation: pulse 1.5s infinite; }
    .step-pending { color: #6c757d; }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Output console */
    .output-console {
        background: #0d1117;
        border-radius: 10px;
        padding: 15px;
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 0.85rem;
        max-height: 350px;
        overflow-y: auto;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_recommendation(score: float) -> tuple:
    """Get recommendation text and color based on score."""
    if score >= 6.5:
        return "✅ Accept", "#00ff88"
    elif score >= 5.5:
        return "🤔 Borderline", "#ffd93d"
    elif score >= 4.5:
        return "👎 Weak Reject", "#ff8c42"
    else:
        return "❌ Reject", "#ff4757"


def render_dimension_bar(score: float, max_score: float, label: str):
    """Render a dimension score bar."""
    percentage = (score / max_score) * 100
    color = "#00ff88" if percentage >= 75 else "#ffd93d" if percentage >= 50 else "#ff4757"
    
    st.markdown(f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="font-weight: 600; color: #c5c6c7;">{label}</span>
            <span style="color: {color}; font-weight: 700;">{score:.0f}/{max_score:.0f}</span>
        </div>
        <div style="background: rgba(255,255,255,0.1); border-radius: 10px; height: 10px; overflow: hidden;">
            <div style="background: {color}; width: {percentage}%; height: 100%; border-radius: 10px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# ASYNC WORKFLOW RUNNER
# =============================================================================

async def run_review_with_streaming(pdf_path: str, venue: str, status_placeholder, output_placeholder):
    """Run the review workflow with real-time streaming updates."""
    
    graph = create_paper_reviewer_graph()
    
    initial_state = {
        "paper_pdf_path": pdf_path,
        "target_venue": venue or "General",
        "paper_markdown": "",
        "paper_metadata": None,
        "validation_passed": False,
        "search_queries": [],
        "search_results": [],
        "related_works": [],
        "selected_related_works": [],
        "review_sections": {},
        "dimension_scores": [],
        "final_score": None,
        "full_review": "",
        "iteration_count": 0,
        "errors": [],
        "current_stage": "start",
        "needs_replanning": False,
    }
    
    config = {"configurable": {"thread_id": f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
    
    # Workflow steps
    steps = [
        ("pdf_to_markdown", "📄 PDF Processing", "Converting PDF to Markdown"),
        ("metadata_extraction", "🔍 Metadata Extraction", "Extracting title, authors, abstract"),
        ("search_query_generation", "🔎 Query Generation", "Creating search queries"),
        ("web_search", "🌐 arXiv Search", "Finding related papers"),
        ("relevance_evaluation", "⚖️ Relevance Scoring", "Evaluating paper relevance"),
        ("reflection", "🤔 Reflection", "Checking progress"),
        ("summarization", "📝 Summarization", "Summarizing related work"),
        ("review_generation", "📋 Review Writing", "Generating peer review"),
        ("dimensional_scoring", "🎯 Scoring", "Calculating final scores"),
    ]
    
    step_status = {s[0]: "pending" for s in steps}
    output_log = []
    all_states = {}
    start_time = datetime.now()
    
    def update_display():
        # Update status
        with status_placeholder.container():
            elapsed = (datetime.now() - start_time).total_seconds()
            st.markdown(f"**⏱️ Elapsed: {elapsed:.0f}s**")
            
            for step_id, step_name, step_desc in steps:
                status = step_status[step_id]
                if status == "complete":
                    st.markdown(f"✅ **{step_name}**")
                elif status == "active":
                    st.markdown(f"🔄 **{step_name}** - _{step_desc}_")
                else:
                    st.markdown(f"⏳ {step_name}")
        
        # Update output log
        with output_placeholder.container():
            log_html = '<div class="output-console">'
            for line in output_log[-12:]:
                if "✅" in line:
                    color = "#00ff88"
                elif "🔄" in line:
                    color = "#00d9ff"
                elif "❌" in line:
                    color = "#ff4757"
                else:
                    color = "#8892b0"
                log_html += f'<div style="color: {color}; margin: 4px 0;">{line}</div>'
            log_html += '</div>'
            st.markdown(log_html, unsafe_allow_html=True)
    
    # Initial log
    output_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 Starting review...")
    output_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📍 Venue: {venue}")
    output_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📄 File: {Path(pdf_path).name}")
    update_display()
    
    try:
        async for state in graph.astream(initial_state, config):
            for node_name, node_state in state.items():
                if isinstance(node_state, dict):
                    all_states.update(node_state)
                    
                    # Log interesting events
                    if node_name == "metadata_extraction" and node_state.get("paper_metadata"):
                        meta = node_state["paper_metadata"]
                        title = meta.title if hasattr(meta, 'title') else meta.get('title', 'Unknown')
                        output_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📌 {title[:40]}...")
                    
                    if node_name == "web_search" and node_state.get("search_results"):
                        n = len(node_state["search_results"])
                        output_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🌐 Found {n} papers")
                    
                    if node_name == "dimensional_scoring" and node_state.get("final_score"):
                        score = node_state["final_score"]
                        output_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🎯 Score: {score:.2f}/10")
                
                # Update step status
                if node_name in step_status:
                    for sid in step_status:
                        if sid == node_name:
                            step_status[sid] = "complete"
                            break
                        elif step_status[sid] == "pending":
                            continue
                    
                    output_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {node_name.replace('_', ' ').title()}")
                    update_display()
                    await asyncio.sleep(0.1)
        
        # Mark all complete
        for sid in step_status:
            step_status[sid] = "complete"
        
        elapsed = (datetime.now() - start_time).total_seconds()
        output_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 Complete! ({elapsed:.1f}s)")
        update_display()
        
        return all_states
        
    except Exception as e:
        output_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Error: {str(e)}")
        update_display()
        raise e


def run_sync(pdf_path: str, venue: str, status_ph, output_ph):
    """Sync wrapper for async workflow."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_review_with_streaming(pdf_path, venue, status_ph, output_ph))
    finally:
        loop.close()


# =============================================================================
# MAIN UI
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 30px 0;">
        <h1 style="font-size: 2.8rem; margin: 0; background: linear-gradient(90deg, #667eea, #764ba2, #e94560); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🎓 Agentic Paper Reviewer
        </h1>
        <p style="color: #8892b0; font-size: 1.1rem; margin-top: 10px;">
            LangGraph-powered AI Research Paper Analysis • Trained on 46,748 ICLR Reviews • Spearman ρ = 0.74
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        venue = st.selectbox(
            "🎯 Target Venue",
            ["ICLR", "NeurIPS", "ICML", "ACL", "CVPR", "AAAI", "General"],
            index=0
        )
        
        st.markdown("---")
        
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #00d9ff;">📊 Scoring Model</h4>
            <p style="color: #8892b0; font-size: 0.9rem;">
                <strong>Data:</strong> 46,748 reviews<br>
                <strong>Papers:</strong> 11,520<br>
                <strong>Spearman ρ:</strong> 0.74
            </p>
            <p style="color: #6c757d; font-size: 0.8rem; margin-top: 10px;">
                <code>-0.31 + 0.71×S + 0.42×P + 1.06×C</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        api_ok = "✅" if os.getenv("OPENAI_API_KEY") else "❌"
        st.markdown(f"**API Status:** {api_ok}")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "📊 Scores", "📝 Full Review", "📚 Related Work"])
    
    # =========================================================================
    # TAB 1: Upload & Process
    # =========================================================================
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📄 Upload Paper")
            
            uploaded_file = st.file_uploader("Drop PDF here", type=["pdf"])
            
            if uploaded_file:
                st.success(f"✅ **{uploaded_file.name}** ({uploaded_file.size/1024/1024:.2f} MB)")
                
                if st.button("🚀 Start Review", use_container_width=True):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        st.session_state.pdf_path = tmp.name
                    st.session_state.review_started = True
                    st.session_state.review_complete = False
        
        with col2:
            st.markdown("### 🔄 Progress")
            status_ph = st.empty()
            output_ph = st.empty()
            
            if not st.session_state.get("review_started"):
                st.markdown("""
                <div style="text-align: center; padding: 50px; color: #6c757d;">
                    <p style="font-size: 3rem;">📄</p>
                    <p>Upload a paper to begin</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Run review if started
        if st.session_state.get("review_started") and not st.session_state.get("review_complete"):
            try:
                result = run_sync(st.session_state.pdf_path, venue, status_ph, output_ph)
                st.session_state.result = result
                st.session_state.review_complete = True
                st.balloons()
                st.success("🎉 Review complete! Check the other tabs for results.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.review_started = False
    
    # =========================================================================
    # TAB 2: Scores Overview
    # =========================================================================
    with tab2:
        if st.session_state.get("review_complete") and st.session_state.get("result"):
            result = st.session_state.result
            final_score = result.get("final_score", 0) or 0
            rec_text, rec_color = get_recommendation(final_score)
            
            # Big score display
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; padding: 30px;">
                    <h2 style="color: #8892b0; margin: 0;">Final Score</h2>
                    <div class="score-display">{final_score:.2f}</div>
                    <p style="font-size: 1.5rem; color: {rec_color}; margin: 10px 0 0 0;">{rec_text}</p>
                </div>
                """, unsafe_allow_html=True)
            
            add_vertical_space(2)
            
            # Dimensional scores
            st.markdown("### 📊 Dimensional Breakdown")
            
            dimensions = result.get("dimension_scores", [])
            cols = st.columns(2)
            
            for i, dim in enumerate(dimensions):
                name = dim.name if hasattr(dim, 'name') else dim.get('name', 'Unknown')
                score = dim.score if hasattr(dim, 'score') else dim.get('score', 0)
                justification = dim.justification if hasattr(dim, 'justification') else dim.get('justification', '')
                max_score = 5 if name == "Confidence" else 4
                
                with cols[i % 2]:
                    with st.expander(f"**{name}**: {score:.0f}/{max_score}", expanded=True):
                        render_dimension_bar(score, max_score, name)
                        st.markdown(f"_{justification}_")
        else:
            st.info("📤 Upload and review a paper to see scores here.")
    
    # =========================================================================
    # TAB 3: Full Review Output (THE MAIN OUTPUT!)
    # =========================================================================
    with tab3:
        if st.session_state.get("review_complete") and st.session_state.get("result"):
            result = st.session_state.result
            review_text = result.get("full_review", "")
            final_score = result.get("final_score", 0) or 0
            rec_text, rec_color = get_recommendation(final_score)
            
            if review_text:
                # Header
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)); border-radius: 15px; padding: 20px; margin-bottom: 25px; border: 1px solid rgba(0, 217, 255, 0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h2 style="margin: 0; color: #00d9ff;">📝 FULL REVIEW</h2>
                            <p style="color: #8892b0; margin: 5px 0 0 0;">Agentic Paper Reviewer Output</p>
                        </div>
                        <div style="text-align: center; background: rgba(0,0,0,0.3); padding: 12px 20px; border-radius: 10px;">
                            <span style="font-size: 2rem; font-weight: 700; color: {rec_color};">{final_score:.2f}</span>
                            <span style="color: #8892b0;">/10</span>
                            <br><span style="color: {rec_color};">{rec_text}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display the FULL review with nice section formatting
                st.markdown("---")
                
                # Section styling
                section_styles = {
                    "Summary": ("📋", "summary", "#00d9ff"),
                    "Strengths": ("✅", "strengths", "#00ff88"),
                    "Weaknesses": ("⚠️", "weaknesses", "#ffd93d"),
                    "Detailed Comments": ("💬", "detailed", "#667eea"),
                    "Questions for Authors": ("❓", "questions", "#ff8c42"),
                    "Missing References": ("📚", "references", "#764ba2"),
                    "Minor Issues": ("🔧", "minor", "#8892b0"),
                    "Recommendation": ("🎯", "recommendation", "#e94560"),
                }
                
                # Parse and display each section
                current_section = None
                section_content = []
                all_sections = {}
                
                for line in review_text.split('\n'):
                    # Check if this is a section header
                    matched_section = None
                    for sec_name in section_styles.keys():
                        if sec_name.lower() in line.lower() and (line.strip().startswith('##') or line.strip().startswith('**')):
                            matched_section = sec_name
                            break
                    
                    if matched_section:
                        if current_section and section_content:
                            all_sections[current_section] = '\n'.join(section_content)
                        current_section = matched_section
                        section_content = []
                    else:
                        section_content.append(line)
                
                # Save last section
                if current_section and section_content:
                    all_sections[current_section] = '\n'.join(section_content)
                
                # Display sections with styling
                if all_sections:
                    for sec_name, (icon, css_class, color) in section_styles.items():
                        if sec_name in all_sections:
                            content = all_sections[sec_name].strip()
                            if content:
                                st.markdown(f"""
                                <div class="review-section {css_class}">
                                    <h3>{icon} {sec_name}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(content)
                                st.markdown("")
                else:
                    # If parsing failed, show raw review
                    st.markdown(review_text)
                
                st.markdown("---")
                
                # Download buttons
                st.markdown("### 📥 Export")
                col1, col2, col3 = st.columns(3)
                
                dimensions = result.get("dimension_scores", [])
                
                with col1:
                    md = f"# Paper Review\n\n**Score: {final_score:.2f}/10** - {rec_text}\n\n"
                    md += "## Scores\n"
                    for d in dimensions:
                        n = d.name if hasattr(d, 'name') else d.get('name')
                        s = d.score if hasattr(d, 'score') else d.get('score')
                        m = 5 if n == "Confidence" else 4
                        md += f"- {n}: {s}/{m}\n"
                    md += f"\n---\n\n{review_text}"
                    
                    st.download_button("📄 Markdown", md, f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", "text/markdown", use_container_width=True)
                
                with col2:
                    js = json.dumps({
                        "score": final_score,
                        "recommendation": rec_text,
                        "dimensions": [{"name": d.name if hasattr(d, 'name') else d.get('name'), "score": d.score if hasattr(d, 'score') else d.get('score')} for d in dimensions],
                        "review": review_text
                    }, indent=2)
                    st.download_button("📊 JSON", js, f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json", use_container_width=True)
                
                with col3:
                    txt = f"PAPER REVIEW\n{'='*60}\nScore: {final_score:.2f}/10 - {rec_text}\n\n{review_text}"
                    st.download_button("📝 Text", txt, f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "text/plain", use_container_width=True)
            else:
                st.warning("No review generated.")
        else:
            st.info("📤 Upload and review a paper to see the full review here.")
    
    # =========================================================================
    # TAB 4: Related Work
    # =========================================================================
    with tab4:
        if st.session_state.get("review_complete") and st.session_state.get("result"):
            result = st.session_state.result
            works = result.get("selected_related_works", [])
            
            if works:
                st.markdown(f"### 📚 Related Work ({len(works)} papers found)")
                
                for i, w in enumerate(works):
                    title = w.title if hasattr(w, 'title') else w.get('title', 'Unknown')
                    arxiv_id = w.arxiv_id if hasattr(w, 'arxiv_id') else w.get('arxiv_id', '')
                    authors = w.authors if hasattr(w, 'authors') else w.get('authors', [])
                    relevance = w.relevance_score if hasattr(w, 'relevance_score') else w.get('relevance_score', 0)
                    abstract = w.abstract if hasattr(w, 'abstract') else w.get('abstract', '')
                    
                    with st.expander(f"**{i+1}. {title[:70]}{'...' if len(title) > 70 else ''}** (Relevance: {relevance:.0%})"):
                        st.markdown(f"**Authors:** {', '.join(authors[:4])}{'...' if len(authors) > 4 else ''}")
                        st.markdown(f"**Abstract:** {abstract[:400]}...")
                        st.link_button("📄 View on arXiv", f"https://arxiv.org/abs/{arxiv_id}")
            else:
                st.info("No related works found.")
        else:
            st.info("📤 Upload and review a paper to see related work here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 15px;">
        Built with ❤️ using LangGraph • Trained on 46,748 ICLR 2025 Reviews • Spearman ρ = 0.74
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INIT
# =============================================================================

if "review_started" not in st.session_state:
    st.session_state.review_started = False
if "review_complete" not in st.session_state:
    st.session_state.review_complete = False
if "result" not in st.session_state:
    st.session_state.result = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()