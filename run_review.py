#!/usr/bin/env python3
"""
Simple CLI Runner for Agentic Paper Reviewer
=============================================

Usage:
    python run_review.py paper.pdf
    python run_review.py paper.pdf --venue ICLR
    python run_review.py paper.pdf --output review.json
    
Requirements:
    pip install -r requirements.txt
    export OPENAI_API_KEY="sk-..."  # or ANTHROPIC_API_KEY
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

# Check for API keys before importing agent
if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
    print("❌ Error: No API key found!")
    print("   Set one of these environment variables:")
    print("   - OPENAI_API_KEY=sk-...")
    print("   - ANTHROPIC_API_KEY=sk-ant-...")
    sys.exit(1)

from agent import create_paper_reviewer_graph, ReviewerState


async def review_paper(
    pdf_path: str,
    venue: str = None,
    output_path: str = None,
    verbose: bool = True
) -> dict:
    """
    Review a paper and return the results.
    
    Args:
        pdf_path: Path to the PDF file
        venue: Target venue (e.g., "ICLR", "NeurIPS", "ICML")
        output_path: Optional path to save JSON output
        verbose: Print progress updates
        
    Returns:
        Dictionary with review results
    """
    # Validate input
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    if not pdf_path.suffix.lower() == '.pdf':
        raise ValueError(f"Expected PDF file, got: {pdf_path.suffix}")
    
    if verbose:
        print("="*60)
        print("🔬 Agentic Paper Reviewer")
        print("="*60)
        print(f"📄 Paper: {pdf_path.name}")
        print(f"🎯 Venue: {venue or 'General'}")
        print("="*60)
    
    # Create the graph
    graph = create_paper_reviewer_graph()
    
    # Initial state
    initial_state = {
        "paper_pdf_path": str(pdf_path),
        "target_venue": venue,
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
    
    # Run the graph
    if verbose:
        print("\n🚀 Starting review process...\n")
    
    config = {"configurable": {"thread_id": f"review_{pdf_path.stem}"}}
    
    # Collect all state updates
    all_states = {}
    async for state in graph.astream(initial_state, config):
        # Merge each node's output into our collected state
        for node_name, node_state in state.items():
            if isinstance(node_state, dict):
                all_states.update(node_state)
            if verbose and node_name != "__end__":
                print(f"  ✓ Completed: {node_name}")
    
    # Use the merged state
    final_state = all_states
    
    # Build result dictionary
    result = {
        "status": "complete" if not final_state.get("errors") else "failed",
        "paper_path": str(pdf_path),
        "venue": venue,
        "metadata": None,
        "review": final_state.get("full_review", ""),
        "scores": {
            "dimensions": [],
            "final_score": final_state.get("final_score"),
        },
        "related_works": [],
        "errors": final_state.get("errors", []),
    }
    
    # Add metadata if available
    if final_state.get("paper_metadata"):
        meta = final_state["paper_metadata"]
        if hasattr(meta, 'model_dump'):
            result["metadata"] = meta.model_dump()
        elif hasattr(meta, 'dict'):
            result["metadata"] = meta.dict()
        else:
            result["metadata"] = meta
    
    # Add dimension scores
    for dim in final_state.get("dimension_scores", []):
        if hasattr(dim, 'model_dump'):
            result["scores"]["dimensions"].append(dim.model_dump())
        elif hasattr(dim, 'dict'):
            result["scores"]["dimensions"].append(dim.dict())
        else:
            result["scores"]["dimensions"].append({
                "name": dim.name,
                "score": dim.score,
                "justification": dim.justification
            })
    
    # Add related works
    for work in final_state.get("selected_related_works", []):
        if hasattr(work, 'model_dump'):
            result["related_works"].append(work.model_dump())
        elif hasattr(work, 'dict'):
            result["related_works"].append(work.dict())
    
    # Print summary
    if verbose:
        print("\n" + "="*60)
        print("📊 REVIEW SUMMARY")
        print("="*60)
        
        if result["metadata"]:
            title = result['metadata'].get('title', 'Unknown')
            print(f"\n📌 Title: {title[:80]}{'...' if len(title) > 80 else ''}")
            authors = result['metadata'].get('authors', [])
            if authors:
                print(f"👥 Authors: {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}")
        
        print(f"\n📈 Dimension Scores:")
        for dim in result["scores"]["dimensions"]:
            name = dim['name']
            score = dim['score']
            if name == "Confidence":
                print(f"   {name:15s}: {score:.0f}/5")
            else:
                print(f"   {name:15s}: {score:.0f}/4")
        
        if result["scores"]["final_score"]:
            score = result["scores"]["final_score"]
            # Add acceptance recommendation based on typical venue thresholds
            if score >= 6.5:
                rec = "✅ Accept"
            elif score >= 5.5:
                rec = "🤔 Borderline"
            elif score >= 4.5:
                rec = "👎 Weak Reject"
            else:
                rec = "❌ Reject"
            print(f"\n🎯 Final Score: {score:.2f}/10 ({rec})")
        
        print(f"\n📚 Related Works Found: {len(result['related_works'])}")
        for i, work in enumerate(result['related_works'][:5]):
            print(f"   {i+1}. {work.get('title', 'Unknown')[:60]}...")
    
    # Save output if requested
    if output_path:
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        if verbose:
            print(f"\n💾 Saved to: {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Review a research paper using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_review.py paper.pdf
    python run_review.py paper.pdf --venue ICLR
    python run_review.py paper.pdf --output review.json --quiet

Environment Variables:
    OPENAI_API_KEY      OpenAI API key (for GPT-4)
    ANTHROPIC_API_KEY   Anthropic API key (for Claude)
        """
    )
    
    parser.add_argument("pdf", help="Path to PDF file to review")
    parser.add_argument("--venue", "-v", 
                        choices=["ICLR", "NeurIPS", "ICML", "ACL", "CVPR", "AAAI", "General"],
                        default=None,
                        help="Target venue for the review")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    
    try:
        result = asyncio.run(review_paper(
            pdf_path=args.pdf,
            venue=args.venue,
            output_path=args.output,
            verbose=not args.quiet
        ))
        
        # Print the full review at the end
        if not args.quiet and result.get("review"):
            print("\n" + "="*60)
            print("📝 FULL REVIEW")
            print("="*60)
            print(result["review"])
        
        # Exit with error code if there were issues
        if result["status"] == "failed":
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()