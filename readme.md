# 🎓 Agentic Paper Reviewer

**AI-powered research paper review system achieving 0.74 Spearman correlation with human peer reviewers.**

Built with LangGraph for multi-agent orchestration and ML-calibrated scoring trained on 46,748 ICLR 2025 reviews.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

- 🔄 **9-Node Agentic Workflow** — LangGraph-based multi-agent system with Plan-Execute-Reflect pattern
- 📊 **ML-Calibrated Scoring** — Regression model trained on 46,748 real peer reviews
- 🔍 **Related Work Search** — Automatic arXiv search with relevance scoring
- 📝 **Structured Reviews** — Summary, Strengths, Weaknesses, Questions, Missing References
- 🎯 **Dimensional Scoring** — Soundness, Presentation, Contribution (matching OpenReview scales)
- 🖥️ **Streamlit UI** — Beautiful interface with real-time progress streaming

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Spearman ρ** | **0.7382** |
| Pearson r | 0.7390 |
| R² Score | 0.5460 |
| Training Reviews | 46,748 |
| Papers | 11,520 |
| vs Prior Work | **+76% improvement** |

---

## 🏗️ Architecture

### LangGraph Workflow

The system uses a 9-node LangGraph workflow with conditional routing and reflection:

<p align="center">
  <img src="docs/graph.png" alt="LangGraph Workflow" width="300"/>
</p>

### System Overview

The architecture combines agentic AI with classical ML for calibrated scoring:

<p align="center">
  <img src="docs/architecture.png" alt="System Architecture" width="800"/>
</p>

**Key Components:**

| Node | Purpose |
|------|---------|
| `pdf_to_markdown` | Convert PDF to structured Markdown using PyMuPDF |
| `metadata_extraction` | Extract title, authors, abstract, keywords |
| `search_query_generation` | Generate multi-specificity search queries |
| `web_search` | Search arXiv for related papers |
| `relevance_evaluation` | Score relevance of found papers |
| `reflection` | Check progress, trigger re-planning if needed |
| `summarization` | Create summaries of relevant papers |
| `review_generation` | Generate comprehensive peer review |
| `dimensional_scoring` | Calculate calibrated final score |

---

## 📐 Scoring Formula

The final score is computed using learned regression weights:

```
Rating = -0.3057 + 0.7134×Soundness + 0.4242×Presentation + 1.0588×Contribution
```

### Learned Weights (Normalized)

| Dimension | Weight | Scale | Description |
|-----------|--------|-------|-------------|
| **Contribution** | 48.2% | 1-4 | Significance and novelty |
| **Soundness** | 32.5% | 1-4 | Technical correctness |
| **Presentation** | 19.3% | 1-4 | Clarity and organization |
| Confidence | — | 1-5 | Reviewer certainty (not in score) |

> 💡 **Key Insight:** Humans reward bold ideas (contribution) almost 2.5x more than clear writing (presentation)!

### Score Interpretation

| Score Range | Recommendation |
|-------------|----------------|
| ≥ 6.5 | ✅ Accept |
| 5.5 - 6.5 | 🤔 Borderline |
| 4.5 - 5.5 | 👎 Weak Reject |
| < 4.5 | ❌ Reject |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/debashis1983/agentic-paper-review.git
cd agentic-paper-reviewer

# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="sk-your-key-here"  # Linux/Mac
set OPENAI_API_KEY=your-key-here       # Windows
```
# For related work search (optional but recommended)
TAVILY_API_KEY=your_tavily_api_key_here

### Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 and upload a paper!

### Run from Command Line

```bash
python run_review.py path/to/paper.pdf --venue ICLR
# Save output to JSON
python agent.py --paper paper.pdf --venue "ICLR 2025" --output review.json
```

---


---

## 📁 Project Structure

```
agentic-paper-reviewer/
├── agent.py              # LangGraph workflow (main agent)
├── streamlit_app.py      # Streamlit UI
├── run_review.py         # CLI interface
├── requirements.txt      # Dependencies
│── train_weights.py     # Regression training script
├── learned_weights.json   # Trained weights
│   
│
├── docs/
│   ├── graph.png         # LangGraph visualization
│   └── architecture.png  # System architecture
│
└── examples/
    └── sample_review.md  # Example output
```

---

## 💻 Usage Examples

### Python API

```python
from agent import AgenticPaperReviewer

reviewer = AgenticPaperReviewer()

result = await reviewer.review_paper(
    paper_path="path/to/paper.pdf",
    target_venue="ICLR"
)

print(f"Score: {result['scores']['final_score']}/10")
print(result['review'])
```

### Review Output Structure

```python
{
    "status": "complete",
    "paper_metadata": {
        "title": "...",
        "authors": ["..."],
        "abstract": "..."
    },
    "review": "## Summary\n...\n## Strengths\n...",
    "scores": {
        "dimensions": [
            {"name": "Soundness", "score": 3, "justification": "..."},
            {"name": "Presentation", "score": 3, "justification": "..."},
            {"name": "Contribution", "score": 3, "justification": "..."},
            {"name": "Confidence", "score": 3, "justification": "..."}
        ],
        "final_score": 6.28,
        "final_score_display": "6.3/10"
    },
    "related_works": [
        {"title": "...", "arxiv_id": "...", "relevance": 0.85}
    ]
}
```

---

## 🔧 Training Your Own Weights

To reproduce the regression training:

```bash
# Download ICLR 2025 reviews from OpenReview
# (Script provided on request)

# Train the regression model
python training/train_weights.py \
    --input iclr_2025.csv \
    --mode individual \
    --confidence exclude

# Output:
# Spearman ρ: 0.7382
# Coefficients: soundness=0.7134, presentation=0.4242, contribution=1.0588
# Intercept: -0.3057
```

---

## 📝 Sample Review Output

```
============================================================
📝 FULL REVIEW
============================================================

## Summary
The paper presents a novel framework for enhancing software agents 
using Monte Carlo Tree Search with iterative refinement...

## Strengths
- **Innovative Integration**: Effectively combines MCTS with LLMs
- **Performance Improvement**: 23% gain on SWE-bench benchmark
- **Comprehensive Framework**: Multi-agent system with clear roles

## Weaknesses
- **Complexity**: Multi-agent overhead may limit applicability
- **Limited Benchmarking**: Only evaluated on SWE-bench

## Questions for Authors
1. How does computational overhead compare to baselines?
2. Are there tasks where the approach performs poorly?

## Recommendation
Accept with minor revisions.

============================================================
📊 DIMENSIONAL SCORES
============================================================
Soundness:     3/4  (Technical claims well-supported)
Presentation:  3/4  (Clear with minor issues)
Contribution:  3/4  (Solid contribution to the field)
Confidence:    3/5  (Fairly confident)

🎯 Final Score: 6.28/10 — 🤔 Borderline
```

---

## 🙏 Acknowledgments

This project was inspired by:

- **Andrew Ng, et al. (Stanford)** Agentic Paper Review
- **LangGraph** by LangChain for the agent orchestration framework
- **OpenReview** for providing access to ICLR peer review data

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@software{agentic_paper_reviewer,
  author = {Debashis Ghosh},
  title = {Agentic Paper Reviewer: ML-Calibrated AI Peer Review},
  year = {2025},
  url = {https://github.com/debashis1983/agentic-paper-review}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

---

<div align="center">

**Built with ❤️ using LangGraph**

[Report Bug](https://github.com/debashis1983/agentic-paper-review/issues) · [Request Feature](https://github.com/debashis1983/agentic-paper-review/issues)

</div>