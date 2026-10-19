# BBN Clinical Empathy Annotation Tool (Enterprise & Research Edition)

An advanced, AI-assisted viewer and annotation system for Breaking Bad News (BBN) clinical conversations, grounded in the **Appraisal Framework for Clinical Empathy** (Lahnala et al., LREC-COLING 2024 / Pounds, 2011) and the **SPIKES Protocol** (Baile et al., 2000).

---

## 🌟 Key Features

- **Interval-Partitioned Span Highlighting**: XSS-safe, overlapping-resilient HTML rendering engine that prevents tag corruption and index shifting.
- **Dual AI Agent Architecture**:
  - **ReAct Agent**: Single-agent reasoning with character coordinate disambiguation and multi-pass JSON extraction.
  - **Multi-Agent Specialist System**: Coordinated specialists (`EO Detector`, `Response Classifier`, `SPIKES Tagger`, and `Relation Linker`).
- **High-Performance SQLite Storage (WAL Mode)**:
  - Atomic **Diff & Sync** state engine: seamlessly handles additions, updates, and deletions with zero zombie annotations.
  - Batch retrieval avoiding N+1 query bottlenecks.
  - Full support for both SQLite database and JSON file backends.
- **Human-in-the-Loop AI Review Tracking**:
  - AI suggestions are recorded with confidence scores and reasoning.
  - Review actions (`accepted`, `rejected`, `modified`) are tracked by annotator/expert.
- **Cross-Turn Semantic Relations**: Link clinician elicitations/responses to patient Empathic Opportunities (EOs) with complete deletion and management support.
- **Occurrence Disambiguation**: Accurately annotates repeated words or phrases without first-occurrence bias.
- **Automated Test Suite**: 100% pass rate across database CRUD, annotation state, XSS safety, and agent parsing.

---

## 📁 Project Structure

```
BBN-Annotation/
├── agents/                      # AI Agent System
│   ├── base.py                  # Core dataclasses, label definitions, AgentConfig
│   ├── prompts.py               # Prompts for ReAct and Specialist Agents
│   ├── react_agent.py           # ReAct Agent with coordinate disambiguation
│   ├── multi_agent.py           # Multi-Agent coordinated system
│   └── runner.py                # Unified CLI runner & expert comparison metrics
├── app/
│   ├── components/              # UI Components
│   │   ├── dialog/              # Annotation Dialog Panels
│   │   │   ├── ai_panel.py      # AI suggestions with review tracking
│   │   │   ├── manual_panel.py  # Manual annotation with occurrence picker
│   │   │   ├── relations_panel.py # Cross-turn relation linking & deletion
│   │   │   └── annotations_panel.py # Current annotations & undo support
│   │   ├── sidebar.py           # Sidebar configuration & conversation selector
│   │   └── turn_card.py         # Safe interval-partitioned turn card rendering
│   ├── database/                # High-performance persistence layer
│   │   ├── connection.py        # SQLite connection with WAL mode & indexes
│   │   ├── models.py            # Typed dataclasses for database entities
│   │   ├── crud.py              # Atomic Diff & Sync and batch operations
│   │   └── migrate.py           # Migration script for sample JSON files
│   ├── services/                # Unified business logic services
│   │   ├── data_service.py      # Unified JSON & SQLite service layer
│   │   ├── annotations.py       # Session CRUD, deduplication, and cascading
│   │   ├── agent.py             # Agent integration & suggestion persistence
│   │   └── conversation.py      # Backward-compatible proxy
│   ├── config.py                # Global settings, color palettes, models
│   ├── state.py                 # Streamlit session state & undo history
│   ├── styles.css               # Modern styling with dark mode support
│   └── main.py                  # Application entry point
├── data/
│   ├── annotation_schema.json   # Formal schema definition
│   └── samples/                 # Sample conversation JSON datasets
├── docs/                        # Theoretical documentation & research papers
├── tests/                       # Comprehensive automated test suite
│   ├── test_database.py         # DB CRUD, batch retrieval, zombie prevention
│   ├── test_annotations.py      # State deduplication & relation cascading
│   ├── test_rendering.py        # XSS escaping & interval partition highlights
│   └── test_agents.py           # Coordinate resolution & JSON extraction
├── requirements.txt             # Project dependencies
├── run.bat                      # Windows double-click launcher
├── run.ps1                      # Windows PowerShell launcher
├── run.sh                       # Linux / macOS shell launcher
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or extract repository
cd BBN-Annotation

# Install dependencies
pip install -r requirements.txt
```

### 2. Setting API Keys (Optional for AI suggestions)

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-api-key"
```

*On Windows Command Prompt:*
```cmd
set OPENAI_API_KEY=your-api-key
```

### 3. Running the Application

- **Windows (Double-Click or Command Prompt):**
  ```cmd
  run.bat
  ```
  *or via PowerShell:*
  ```powershell
  .\run.ps1
  ```

- **Linux / macOS:**
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

- **Manual Streamlit Launch:**
  ```bash
  streamlit run app/main.py --server.port 8501
  ```

Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the complete test suite with Pytest:

```bash
python -m pytest tests -v
```

All 51 automated unit and integration tests validate:
- **Maximum Weight Bipartite Matching**: 1-to-1 Hungarian matching algorithm resolving ambiguous assignments, identical labels, and Cartesian IoU pollution.
- **Data Model Serialization**: Full round-trip integrity (`to_dict` / `from_dict`), explicit reasoning field support, and backward compatibility.
- **Transactional Database Layer**: WAL mode, foreign-key cascades, atomic rollback on exceptions, duplicate rejection, and NULL-expert SPIKES upsert integrity.
- **HTML & XSS Security**: Robust sanitization across turn text, dialog boxes, chips, relations, and interval-partitioned highlights.
- **Deterministic Mocked Agents**: Multi-pass JSON extraction, Markdown code fence parsing, bounded retries, and provider error reporting.

---

## 📐 Evaluation Methodology: 1-to-1 Maximum Bipartite Matching

The expert evaluation pipeline (`AnnotationRunner.compare_with_expert`) utilizes **Maximum Weight Bipartite Matching (Hungarian algorithm)**:
- **One-to-One Alignment**: Each predicted span matches at most one ground-truth span with matching label.
- **Cardinality Maximization**: Matches cardinal count ($TP$) first using high-order weight scaling before maximizing total IoU, preventing greedy assignment suboptimalities.
- **Overlap & Exact Matching**: Supports both continuous IoU thresholds (`min_iou_threshold`) and exact boundary match (`exact_match=True`).
- **Safe Zero Denominators**: Fully handles empty inputs and edge cases without division-by-zero, NaN, or infinite metric values.

---

## ⚕️ Clinical & Scientific Validity Disclaimer

> [!IMPORTANT]
> **Research Assistance Disclaimer**:
> This software is an experimental research and annotation tool designed to assist human experts in studying clinical dialogue. Model-generated annotations, suggestions, and agreement metrics are **research aids and do NOT constitute clinical validation or medical diagnosis**.
> 
> - **Synthetic Demo Data**: Sample conversations included in `data/samples/` are synthetic dialogues constructed for demonstration and user-interface testing. They do not contain protected health information (PHI).
> - **Inter-Annotator Agreement**: Scientific evaluation of clinical empathy requires qualified expert adjudication and representative real-world cohorts. Unit test passes demonstrate software correctness, not clinical efficacy.

---

## 📚 References & Scientific Background

1. **Lahnala, A., Neuendorf, B., Thomin, A., Welch, C., Stibane, T., & Flek, L. (2024)**. *Appraisal Framework for Clinical Empathy: A Novel Application to Breaking Bad News Conversations*. In Proceedings of LREC-COLING 2024.
2. **Pounds, G. (2011)**. *Appraisal Framework for Clinical Empathy*.
3. **Baile, W. F., et al. (2000)**. *SPIKES—A Six-Step Protocol for Delivering Bad News: Application to the Patient with Cancer*. The Oncologist, 5(4), 302-311.
