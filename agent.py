"""
LangGraph-based Agentic Paper Reviewer
======================================


This agent provides rapid, actionable feedback on research papers by:
1. Converting PDF to Markdown
2. Extracting paper metadata and validating it's academic
3. Generating multi-specificity search queries
4. Finding and analyzing related work from arXiv
5. Synthesizing comprehensive reviews with dimensional scoring

Architecture follows the Planner-Executor-Critic pattern with LangGraph.
"""

import os
import re
import json
import asyncio
import operator
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field
from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

class ReviewDimension(BaseModel):
    """Scoring dimension for paper review."""
    name: str
    score: float = Field(ge=1, le=10)
    justification: str


class RelatedWork(BaseModel):
    """Related work metadata and summary."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    relevance_score: float = Field(ge=0, le=1)
    summary_type: Literal["abstract", "detailed"] = "abstract"
    detailed_summary: Optional[str] = None
    focus_areas: List[str] = []


class PaperMetadata(BaseModel):
    """Extracted paper metadata."""
    title: str
    authors: List[str] = []
    abstract: str = ""
    is_academic_paper: bool = True
    venue: Optional[str] = None
    keywords: List[str] = []


class ReviewerState(TypedDict):
    """Main state for the agentic reviewer workflow."""
    # Input
    paper_pdf_path: str
    target_venue: Optional[str]
    
    # Processing stages
    paper_markdown: str
    paper_metadata: Optional[PaperMetadata]
    validation_passed: bool
    
    # Search and related work
    search_queries: List[Dict[str, str]]  # {query, specificity, perspective}
    search_results: List[Dict[str, Any]]
    related_works: List[RelatedWork]
    selected_related_works: List[RelatedWork]
    
    # Review generation
    review_sections: Dict[str, str]
    dimension_scores: List[ReviewDimension]
    final_score: Optional[float]
    
    # Output
    full_review: str
    
    # Workflow control
    iteration_count: int
    errors: Annotated[List[str], operator.add]
    current_stage: str
    needs_replanning: bool


# =============================================================================
# PROMPTS
# =============================================================================

METADATA_EXTRACTION_PROMPT = """You are an expert at analyzing academic papers.

Given the following paper content in Markdown format, extract:
1. Title
2. Authors (list of names)
3. Abstract
4. Whether this is an academic research paper (not a blog post, tutorial, etc.)
5. Keywords/topics

Paper content:
{paper_content}

Respond in JSON format:
{{
    "title": "...",
    "authors": ["...", "..."],
    "abstract": "...",
    "is_academic_paper": true/false,
    "keywords": ["...", "..."]
}}
"""

SEARCH_QUERY_GENERATION_PROMPT = """You are an expert research assistant helping to find related work for a paper review.

Paper Title: {title}
Abstract: {abstract}
Keywords: {keywords}

Generate 6-8 search queries to find relevant prior work on arXiv. Include queries at different specificity levels:

1. HIGH SPECIFICITY (2-3 queries): Very specific to the exact method/approach
2. MEDIUM SPECIFICITY (2-3 queries): Related techniques and benchmarks
3. LOW SPECIFICITY (2-3 queries): Broader problem domain and applications

For each query, also specify the PERSPECTIVE:
- "baseline": Find papers that could be baselines/comparisons
- "method": Find papers with similar methods/techniques
- "problem": Find papers addressing the same problem
- "benchmark": Find relevant benchmarks/datasets

Respond in JSON format:
{{
    "queries": [
        {{"query": "...", "specificity": "high/medium/low", "perspective": "baseline/method/problem/benchmark"}},
        ...
    ]
}}
"""

RELEVANCE_EVALUATION_PROMPT = """You are evaluating the relevance of a potential related work for a paper review.

Paper being reviewed:
Title: {paper_title}
Abstract: {paper_abstract}

Candidate related work:
Title: {candidate_title}
Abstract: {candidate_abstract}

Rate the relevance from 0.0 to 1.0 and decide the summarization approach:
- If relevance >= 0.7: Consider "detailed" summary (requires downloading full paper)
- If relevance >= 0.4: Use "abstract" summary (use existing abstract)
- If relevance < 0.4: May be excluded

If detailed summary is needed, specify focus areas (what aspects to emphasize).

Respond in JSON format:
{{
    "relevance_score": 0.0-1.0,
    "summary_type": "abstract" or "detailed",
    "focus_areas": ["...", "..."],
    "reasoning": "..."
}}
"""

DETAILED_SUMMARY_PROMPT = """You are creating a focused summary of a research paper for use in a review.

Paper to summarize:
{paper_content}

Focus areas to emphasize:
{focus_areas}

Context - this summary will be used to review another paper titled:
"{review_paper_title}"

Create a detailed summary (300-500 words) that:
1. Captures the main contributions
2. Emphasizes the specified focus areas
3. Notes methodology and key results
4. Highlights aspects relevant to the paper being reviewed

Summary:
"""

REVIEW_GENERATION_PROMPT = """You are an expert peer reviewer providing constructive feedback on a research paper.

=== PAPER BEING REVIEWED ===
{paper_content}

=== RELATED WORK CONTEXT ===
{related_work_summaries}

=== TARGET VENUE ===
{target_venue}

Generate a comprehensive peer review following this structure:

## Summary
Brief summary of the paper's main contributions and approach.

## Strengths
- List key strengths with specific examples from the paper

## Weaknesses
- List weaknesses with constructive suggestions for improvement

## Detailed Comments
Provide specific, actionable feedback organized by section.

## Questions for Authors
List clarifying questions that would help improve the paper.

## Missing References
Based on the related work, note any important missing citations.

## Minor Issues
Grammar, formatting, clarity issues.

## Recommendation
Overall assessment and recommendation.

Provide your review:
"""

DIMENSIONAL_SCORING_PROMPT = """You are scoring a research paper on 4 core dimensions used by major ML venues (NeurIPS, ICLR, ICML).

=== PAPER ===
{paper_content}

=== YOUR REVIEW ===
{review_content}

Score each dimension using the EXACT scales used by OpenReview:

1. **Soundness** (1-4 scale: Technical Quality)
   - Are the claims well-supported by theoretical analysis or empirical evidence?
   - Is the methodology rigorous and appropriate?
   - Are the experiments well-designed with proper baselines and ablations?
   Scale:
     1 = Poor: Major technical flaws
     2 = Fair: Some concerns about correctness
     3 = Good: Technically sound with minor issues
     4 = Excellent: Technically excellent, no concerns

2. **Presentation** (1-4 scale: Clarity & Organization)
   - Is the paper well-written and easy to follow?
   - Are figures, tables, and equations clear and informative?
   - Is the paper well-organized?
   Scale:
     1 = Poor: Hard to follow, poorly written
     2 = Fair: Some clarity issues
     3 = Good: Clear with minor issues
     4 = Excellent: Exceptionally clear and well-organized

3. **Contribution** (1-4 scale: Significance & Novelty)
   - How significant is this work to the field?
   - Does it provide new insights, methods, or results?
   - How does it advance the state of the art?
   Scale:
     1 = Poor: Little to no contribution
     2 = Fair: Incremental or limited contribution
     3 = Good: Solid contribution
     4 = Excellent: Groundbreaking contribution

4. **Confidence** (1-5 scale: Your Assessment Certainty)
   - How confident are you in your evaluation?
   - How familiar are you with the related work?
   Scale:
     1 = Low: Educated guess
     2 = Somewhat confident
     3 = Fairly confident: Familiar with the area
     4 = Confident: Checked key claims
     5 = Very confident: Absolutely certain, expert in this area

Respond in JSON format:
{{
    "dimensions": [
        {{"name": "Soundness", "score": X, "justification": "..."}},
        {{"name": "Presentation", "score": X, "justification": "..."}},
        {{"name": "Contribution", "score": X, "justification": "..."}},
        {{"name": "Confidence", "score": X, "justification": "..."}}
    ]
}}

IMPORTANT: Use the exact scales above (Soundness/Presentation/Contribution: 1-4, Confidence: 1-5).
"""


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

def get_llm(model: str = "gpt-4o", temperature: float = 0.3):
    """Get LLM instance based on available API keys."""
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model=model, temperature=temperature)
    else:
        raise ValueError("No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")


# =============================================================================
# NODE IMPLEMENTATIONS
# =============================================================================

async def pdf_to_markdown_node(state: ReviewerState) -> dict:
    """Convert PDF to Markdown using document extraction."""
    print("📄 Converting PDF to Markdown...")
    
    try:
        
        pdf_path = state["paper_pdf_path"]
        
        # Check if it's already markdown content (for testing)
        if pdf_path.endswith(".md") or not os.path.exists(pdf_path):
            # Assume it's already markdown or use provided content
            if "paper_markdown" in state and state["paper_markdown"]:
                return {"paper_markdown": state["paper_markdown"], "current_stage": "metadata_extraction"}
        
        # Try to extract with available tools
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            markdown_content = ""
            for page in doc:
                markdown_content += page.get_text("text") + "\n\n"
            doc.close()
        except ImportError:
            # Fallback: read as text if possible
            with open(pdf_path, 'r', encoding='utf-8', errors='ignore') as f:
                markdown_content = f.read()
        
        return {
            "paper_markdown": markdown_content,
            "current_stage": "metadata_extraction"
        }
    
    except Exception as e:
        return {
            "errors": [f"PDF conversion failed: {str(e)}"],
            "current_stage": "failed"
        }


async def metadata_extraction_node(state: ReviewerState) -> dict:
    """Extract paper metadata and validate it's academic."""
    print("🔍 Extracting paper metadata...")
    
    try:
        llm = get_llm()
        
        # Truncate content if too long
        content = state["paper_markdown"][:15000]
        
        prompt = METADATA_EXTRACTION_PROMPT.format(paper_content=content)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        # Parse JSON response
        response_text = response.content
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            metadata_dict = json.loads(json_match.group())
            metadata = PaperMetadata(**metadata_dict)
        else:
            raise ValueError("Could not parse metadata response")
        
        validation_passed = metadata.is_academic_paper
        
        if not validation_passed:
            return {
                "paper_metadata": metadata,
                "validation_passed": False,
                "errors": ["Document does not appear to be an academic paper"],
                "current_stage": "failed"
            }
        
        return {
            "paper_metadata": metadata,
            "validation_passed": True,
            "current_stage": "search_query_generation"
        }
    
    except Exception as e:
        return {
            "errors": [f"Metadata extraction failed: {str(e)}"],
            "current_stage": "failed"
        }


async def search_query_generation_node(state: ReviewerState) -> dict:
    """Generate diverse search queries for finding related work."""
    print("🔎 Generating search queries...")
    
    try:
        llm = get_llm()
        metadata = state["paper_metadata"]
        
        prompt = SEARCH_QUERY_GENERATION_PROMPT.format(
            title=metadata.title,
            abstract=metadata.abstract,
            keywords=", ".join(metadata.keywords)
        )
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        # Parse JSON response
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        if json_match:
            queries_data = json.loads(json_match.group())
            queries = queries_data.get("queries", [])
        else:
            # Fallback: generate basic queries
            queries = [
                {"query": metadata.title, "specificity": "high", "perspective": "method"},
                {"query": f"{metadata.keywords[0] if metadata.keywords else 'deep learning'}", 
                 "specificity": "medium", "perspective": "problem"}
            ]
        
        return {
            "search_queries": queries,
            "current_stage": "web_search"
        }
    
    except Exception as e:
        return {
            "errors": [f"Query generation failed: {str(e)}"],
            "search_queries": [],
            "current_stage": "web_search"
        }


async def web_search_node(state: ReviewerState) -> dict:
    """Execute search queries to find related papers on arXiv."""
    print("🌐 Searching for related work...")
    
    try:
        # In production, use Tavily API or similar
        # For demonstration, we'll simulate with arXiv API
        
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET
        
        all_results = []
        
        for query_info in state["search_queries"][:5]:  # Limit to 5 queries
            query = query_info["query"]
            
            # arXiv API search
            base_url = "http://export.arxiv.org/api/query?"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": 5
            }
            url = base_url + urllib.parse.urlencode(params)
            
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    xml_data = response.read().decode('utf-8')
                
                # Parse XML response
                root = ET.fromstring(xml_data)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                
                for entry in root.findall('atom:entry', ns):
                    title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                    abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                    arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
                    
                    authors = []
                    for author in entry.findall('atom:author', ns):
                        name = author.find('atom:name', ns).text
                        authors.append(name)
                    
                    all_results.append({
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "authors": authors,
                        "abstract": abstract[:1000],
                        "query_source": query_info
                    })
            
            except Exception as search_error:
                print(f"  Search error for '{query}': {search_error}")
                continue
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.5)
        
        # Deduplicate by arxiv_id
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result["arxiv_id"] not in seen_ids:
                seen_ids.add(result["arxiv_id"])
                unique_results.append(result)
        
        return {
            "search_results": unique_results,
            "current_stage": "relevance_evaluation"
        }
    
    except Exception as e:
        return {
            "errors": [f"Web search failed: {str(e)}"],
            "search_results": [],
            "current_stage": "relevance_evaluation"
        }


async def relevance_evaluation_node(state: ReviewerState) -> dict:
    """Evaluate relevance of search results and select top papers."""
    print("⚖️ Evaluating relevance of related works...")
    
    try:
        llm = get_llm()
        metadata = state["paper_metadata"]
        related_works = []
        
        for result in state["search_results"][:10]:  # Evaluate top 10
            prompt = RELEVANCE_EVALUATION_PROMPT.format(
                paper_title=metadata.title,
                paper_abstract=metadata.abstract,
                candidate_title=result["title"],
                candidate_abstract=result["abstract"]
            )
            
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            
            try:
                json_match = re.search(r'\{[\s\S]*\}', response.content)
                if json_match:
                    eval_data = json.loads(json_match.group())
                    
                    related_work = RelatedWork(
                        arxiv_id=result["arxiv_id"],
                        title=result["title"],
                        authors=result["authors"],
                        abstract=result["abstract"],
                        relevance_score=eval_data.get("relevance_score", 0.5),
                        summary_type=eval_data.get("summary_type", "abstract"),
                        focus_areas=eval_data.get("focus_areas", [])
                    )
                    related_works.append(related_work)
            
            except Exception as parse_error:
                # Include with default relevance
                related_work = RelatedWork(
                    arxiv_id=result["arxiv_id"],
                    title=result["title"],
                    authors=result["authors"],
                    abstract=result["abstract"],
                    relevance_score=0.5,
                    summary_type="abstract"
                )
                related_works.append(related_work)
        
        # Sort by relevance and select top
        related_works.sort(key=lambda x: x.relevance_score, reverse=True)
        selected = related_works[:7]  # Top 7 most relevant
        
        return {
            "related_works": related_works,
            "selected_related_works": selected,
            "current_stage": "summarization"
        }
    
    except Exception as e:
        return {
            "errors": [f"Relevance evaluation failed: {str(e)}"],
            "related_works": [],
            "selected_related_works": [],
            "current_stage": "summarization"
        }


async def summarization_node(state: ReviewerState) -> dict:
    """Generate summaries for selected related works."""
    print("📝 Summarizing related works...")
    
    try:
        llm = get_llm()
        metadata = state["paper_metadata"]
        updated_works = []
        
        for work in state["selected_related_works"]:
            if work.summary_type == "detailed" and work.focus_areas:
                # In production: download paper PDF and create detailed summary
                # For now, create enhanced summary from abstract
                
                prompt = DETAILED_SUMMARY_PROMPT.format(
                    paper_content=f"Title: {work.title}\n\nAbstract: {work.abstract}",
                    focus_areas=", ".join(work.focus_areas),
                    review_paper_title=metadata.title
                )
                
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                work.detailed_summary = response.content
            
            updated_works.append(work)
        
        return {
            "selected_related_works": updated_works,
            "current_stage": "review_generation"
        }
    
    except Exception as e:
        return {
            "errors": [f"Summarization failed: {str(e)}"],
            "current_stage": "review_generation"
        }


async def review_generation_node(state: ReviewerState) -> dict:
    """Generate comprehensive paper review."""
    print("📋 Generating review...")
    
    try:
        llm = get_llm(temperature=0.4)
        
        # Prepare related work summaries
        related_summaries = []
        for i, work in enumerate(state["selected_related_works"], 1):
            summary = work.detailed_summary or work.abstract
            related_summaries.append(
                f"[{i}] {work.title} (arXiv:{work.arxiv_id})\n"
                f"Authors: {', '.join(work.authors[:3])}{'...' if len(work.authors) > 3 else ''}\n"
                f"Relevance: {work.relevance_score:.2f}\n"
                f"Summary: {summary[:500]}..."
            )
        
        related_work_text = "\n\n".join(related_summaries) if related_summaries else "No related works found."
        
        # Truncate paper content for context window
        paper_content = state["paper_markdown"][:20000]
        
        prompt = REVIEW_GENERATION_PROMPT.format(
            paper_content=paper_content,
            related_work_summaries=related_work_text,
            target_venue=state.get("target_venue", "General ML/AI venue")
        )
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        return {
            "full_review": response.content,
            "current_stage": "dimensional_scoring"
        }
    
    except Exception as e:
        return {
            "errors": [f"Review generation failed: {str(e)}"],
            "current_stage": "dimensional_scoring"
        }


async def dimensional_scoring_node(state: ReviewerState) -> dict:
    """Score paper on 7 dimensions for final score calculation."""
    print("🎯 Calculating dimensional scores...")
    
    try:
        llm = get_llm(temperature=0.2)
        
        prompt = DIMENSIONAL_SCORING_PROMPT.format(
            paper_content=state["paper_markdown"][:15000],
            review_content=state["full_review"]
        )
        
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        # Parse scores
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        if json_match:
            scores_data = json.loads(json_match.group())
            dimensions = [
                ReviewDimension(**dim) 
                for dim in scores_data.get("dimensions", [])
            ]
        else:
            dimensions = []
        
        # Calculate final score using learned weights from regression
        # 
        # =====================================================================
        # TRAINED ON: 46,748 ICLR 2025 reviews (11,520 papers)
        # MODEL PERFORMANCE:
        #   - Spearman ρ = 0.7382
        #   - R² Score  = 0.5460
        #   - Pearson r = 0.7390
        #   - RMSE      = 1.1439
        # =====================================================================
        #
        # SCALES (matching OpenReview exactly):
        #   - Soundness: 1-4
        #   - Presentation: 1-4  
        #   - Contribution: 1-4
        #   - Confidence: 1-5 (not used in final score)
        #   - Rating (output): 1-10
        #
        # REGRESSION FORMULA:
        #   rating = -0.3057 + 0.7134×soundness + 0.4242×presentation + 1.0588×contribution
        #
        # NORMALIZED WEIGHTS:
        #   - Soundness:    32.5%
        #   - Presentation: 19.3%
        #   - Contribution: 48.2%
        # =====================================================================
        
        # Regression coefficients from ICLR 2025 training (46,748 reviews)
        INTERCEPT = -0.3057
        COEF_SOUNDNESS = 0.7134       # 32.5% normalized
        COEF_PRESENTATION = 0.4242    # 19.3% normalized
        COEF_CONTRIBUTION = 1.0588    # 48.2% normalized
        
        if dimensions:
            # Extract scores from dimensions
            scores = {}
            confidence_score = None
            
            for dim in dimensions:
                if dim.name == "Confidence":
                    confidence_score = dim.score
                else:
                    scores[dim.name] = dim.score
            
            # Get dimension scores (already on 1-4 scale from LLM)
            soundness = scores.get("Soundness", 2.5)
            presentation = scores.get("Presentation", 2.5)
            contribution = scores.get("Contribution", 2.5)
            
            # Apply regression formula with intercept
            final_score = (
                INTERCEPT +
                COEF_SOUNDNESS * soundness +
                COEF_PRESENTATION * presentation +
                COEF_CONTRIBUTION * contribution
            )
            
            # Clamp to valid range (1-10)
            final_score = max(1.0, min(10.0, final_score))
            
            # Log details for transparency
            if confidence_score:
                print(f"  📊 Reviewer confidence: {confidence_score}/5")
            
            # Show calculation breakdown
            print(f"  📐 Score: {INTERCEPT:.2f} + {COEF_SOUNDNESS}×{soundness:.0f} + "
                  f"{COEF_PRESENTATION}×{presentation:.0f} + {COEF_CONTRIBUTION}×{contribution:.0f} "
                  f"= {final_score:.2f}")
        else:
            final_score = None
        
        return {
            "dimension_scores": dimensions,
            "final_score": final_score,
            "current_stage": "complete"
        }
    
    except Exception as e:
        return {
            "errors": [f"Scoring failed: {str(e)}"],
            "dimension_scores": [],
            "final_score": None,
            "current_stage": "complete"
        }


async def reflection_node(state: ReviewerState) -> dict:
    """Reflect on progress and decide if replanning is needed."""
    print("🤔 Reflecting on progress...")
    
    iteration = state.get("iteration_count", 0) + 1
    needs_replanning = False
    
    # Check for issues that might require replanning
    if not state.get("search_results") and state.get("current_stage") == "relevance_evaluation":
        needs_replanning = True
        print("  ⚠️ No search results - may need to broaden queries")
    
    if len(state.get("selected_related_works", [])) < 3 and state.get("current_stage") == "summarization":
        needs_replanning = True
        print("  ⚠️ Few related works found - may need additional searches")
    
    return {
        "iteration_count": iteration,
        "needs_replanning": needs_replanning
    }


# =============================================================================
# CONDITIONAL EDGES
# =============================================================================

def route_after_validation(state: ReviewerState) -> str:
    """Route based on validation results."""
    if state.get("validation_passed"):
        return "search_query_generation"
    return END


def route_after_reflection(state: ReviewerState) -> str:
    """Route based on reflection results."""
    if state.get("needs_replanning") and state.get("iteration_count", 0) < 3:
        return "search_query_generation"  # Replan
    return "continue"


def route_by_stage(state: ReviewerState) -> str:
    """Route based on current stage."""
    stage = state.get("current_stage", "")
    
    if stage == "failed":
        return END
    elif stage == "complete":
        return END
    
    stage_map = {
        "metadata_extraction": "metadata_extraction",
        "search_query_generation": "search_query_generation",
        "web_search": "web_search",
        "relevance_evaluation": "relevance_evaluation",
        "summarization": "summarization",
        "review_generation": "review_generation",
        "dimensional_scoring": "dimensional_scoring"
    }
    
    return stage_map.get(stage, END)


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_paper_reviewer_graph():
    """Create the LangGraph workflow for paper review."""
    
    # Initialize graph with state schema
    workflow = StateGraph(ReviewerState)
    
    # Add nodes
    workflow.add_node("pdf_to_markdown", pdf_to_markdown_node)
    workflow.add_node("metadata_extraction", metadata_extraction_node)
    workflow.add_node("search_query_generation", search_query_generation_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("relevance_evaluation", relevance_evaluation_node)
    workflow.add_node("summarization", summarization_node)
    workflow.add_node("review_generation", review_generation_node)
    workflow.add_node("dimensional_scoring", dimensional_scoring_node)
    workflow.add_node("reflection", reflection_node)
    
    # Add edges - main flow
    workflow.add_edge(START, "pdf_to_markdown")
    workflow.add_edge("pdf_to_markdown", "metadata_extraction")
    
    # Conditional after validation
    workflow.add_conditional_edges(
        "metadata_extraction",
        route_after_validation,
        {
            "search_query_generation": "search_query_generation",
            END: END
        }
    )
    
    workflow.add_edge("search_query_generation", "web_search")
    workflow.add_edge("web_search", "relevance_evaluation")
    workflow.add_edge("relevance_evaluation", "reflection")
    
    # Conditional after reflection (every 3rd iteration check)
    workflow.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "search_query_generation": "search_query_generation",
            "continue": "summarization"
        }
    )
    
    workflow.add_edge("summarization", "review_generation")
    workflow.add_edge("review_generation", "dimensional_scoring")
    workflow.add_edge("dimensional_scoring", END)
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# =============================================================================
# MAIN INTERFACE
# =============================================================================

class AgenticPaperReviewer:
    """Main interface for the agentic paper reviewer."""
    
    def __init__(self):
        self.graph = create_paper_reviewer_graph()
    
    async def review_paper(
        self,
        paper_path: str = None,
        paper_content: str = None,
        target_venue: str = None,
        thread_id: str = None
    ) -> Dict[str, Any]:
        """
        Review a paper and generate comprehensive feedback.
        
        Args:
            paper_path: Path to PDF file
            paper_content: Raw markdown/text content (alternative to path)
            target_venue: Target venue (e.g., "ICLR", "NeurIPS", "ACL")
            thread_id: Thread ID for checkpoint persistence
        
        Returns:
            Dictionary containing review, scores, and metadata
        """
        
        if not paper_path and not paper_content:
            raise ValueError("Must provide either paper_path or paper_content")
        
        initial_state: ReviewerState = {
            "paper_pdf_path": paper_path or "",
            "target_venue": target_venue,
            "paper_markdown": paper_content or "",
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
            "current_stage": "pdf_to_markdown" if paper_path else "metadata_extraction",
            "needs_replanning": False
        }
        
        config = {"configurable": {"thread_id": thread_id or f"review_{datetime.now().isoformat()}"}}
        
        # Execute workflow
        final_state = await self.graph.ainvoke(initial_state, config)
        
        # Format output
        return self._format_output(final_state)
    
    def _format_output(self, state: ReviewerState) -> Dict[str, Any]:
        """Format the final output."""
        
        output = {
            "status": "complete" if state.get("full_review") else "failed",
            "paper_metadata": {
                "title": state["paper_metadata"].title if state.get("paper_metadata") else "Unknown",
                "authors": state["paper_metadata"].authors if state.get("paper_metadata") else [],
                "abstract": state["paper_metadata"].abstract[:500] if state.get("paper_metadata") else ""
            } if state.get("paper_metadata") else None,
            "review": state.get("full_review", ""),
            "scores": {
                "dimensions": [
                    {"name": d.name, "score": d.score, "justification": d.justification}
                    for d in state.get("dimension_scores", [])
                ],
                "final_score": state.get("final_score"),
                "final_score_display": f"{state['final_score']:.1f}/10" if state.get("final_score") else None
            },
            "related_works": [
                {
                    "title": w.title,
                    "arxiv_id": w.arxiv_id,
                    "relevance": w.relevance_score,
                    "url": f"https://arxiv.org/abs/{w.arxiv_id}"
                }
                for w in state.get("selected_related_works", [])
            ],
            "metadata": {
                "target_venue": state.get("target_venue"),
                "iterations": state.get("iteration_count", 0),
                "errors": state.get("errors", [])
            }
        }
        
        return output


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic Paper Reviewer")
    parser.add_argument("--paper", "-p", help="Path to paper PDF or markdown file")
    parser.add_argument("--venue", "-v", help="Target venue (e.g., ICLR, NeurIPS)")
    parser.add_argument("--output", "-o", help="Output file for review (JSON)")
    
    args = parser.parse_args()
    
    if not args.paper:
        # Demo mode with sample content
        print("🎓 Agentic Paper Reviewer - Demo Mode")
        print("=" * 50)
        
        sample_content = """
        # Sample Paper: Attention Is All You Need (Demo)
        
        ## Abstract
        We propose a new simple network architecture, the Transformer, based solely 
        on attention mechanisms, dispensing with recurrence and convolutions entirely.
        Our model achieves state-of-the-art results on machine translation tasks.
        
        ## Introduction
        Recurrent neural networks have been the dominant approach for sequence modeling.
        However, they suffer from sequential computation that prevents parallelization.
        
        ## Method
        We introduce the Transformer architecture with multi-head self-attention.
        
        ## Experiments
        We evaluate on WMT English-German and English-French translation.
        Our model achieves 28.4 BLEU on English-German.
        
        ## Conclusion
        The Transformer architecture offers a promising new paradigm for sequence modeling.
        """
        
        reviewer = AgenticPaperReviewer()
        result = await reviewer.review_paper(
            paper_content=sample_content,
            target_venue="NeurIPS"
        )
        
        print("\n📋 REVIEW OUTPUT")
        print("=" * 50)
        print(json.dumps(result, indent=2, default=str))
    
    else:
        reviewer = AgenticPaperReviewer()
        
        # Check if file is markdown or PDF
        if args.paper.endswith('.md') or args.paper.endswith('.txt'):
            with open(args.paper, 'r') as f:
                content = f.read()
            result = await reviewer.review_paper(
                paper_content=content,
                target_venue=args.venue
            )
        else:
            result = await reviewer.review_paper(
                paper_path=args.paper,
                target_venue=args.venue
            )
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"✅ Review saved to {args.output}")
        else:
            print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())