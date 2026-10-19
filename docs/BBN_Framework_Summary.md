# BBN (Breaking Bad News) Empathy Framework Summary

## Overview

The BBN Empathy Framework is a novel annotation scheme for clinical empathy communication in patient-clinician conversations, specifically designed for Breaking Bad News (BBN) scenarios.

---

## Authors and Publication

**Creators:**
- Allison Lahnala
- Béla Neuendorf
- Alexander Thomin
- Charles Welch
- Tina Stibane
- Lucie Flek

**Publication:** "Appraisal Framework for Clinical Empathy: A Novel Application to Breaking Bad News Conversations" - LREC-COLING 2024

**GitHub Repository:** https://github.com/caisa-lab/BBN-Empathy

**Collaboration:** Developed with a large German medical school for medical didactics research.

---

## Theoretical Foundations

The framework combines two established theories:

### 1. Pounds' Appraisal Framework for Clinical Empathy

Based on **Systemic Functional Linguistics (SFL)** and extends Suchman et al.'s (1997) model of clinical empathic communication. It categorizes discourse elements along three attitudinal dimensions:

- **Feelings (Affect)** - emotions, emotive behaviors, mental states
- **Judgement** - attitudes/perceptions toward self or others
- **Appreciation** - attitudes/perceptions toward things, events, actions, and behaviors

### 2. SPIKES Protocol

A 6-step pedagogical framework commonly taught in medical didactics for BBN conversations:

| Step | Description |
|------|-------------|
| **S**etting | Establishing appropriate time, place, and environment for the conversation |
| **P**erception | Understanding patient's current knowledge and perception of their situation |
| **I**nvitation | Assessing how much information the patient is ready to receive |
| **K**nowledge | Delivering the medical information clearly and compassionately |
| **E**mpathy/Emotion | Responding empathetically to patient reactions |
| **S**trategy and Summary | Discussing treatment plans if patient is ready |

---

## Annotation Structure

### Patient Role: Empathic Opportunities (EOs)

Empathic opportunities are expressions or behaviors that reveal a patient's feelings or views, which can be either **explicit** or **implicit**.

| Type | Explicit | Implicit |
|------|----------|----------|
| **Feelings** | Describing or exhibiting emotion quality ("I'm sad"), emotive behavior ("I cried"), or mental state ("I'm in pain") | Implicitly expressing feeling by referring to negative experiences or critical life stages ("My aunt had the same condition...") |
| **Appreciation** | Direct attitude toward things/events ("The MRI was boring", "The medication is not helping me") | Indirectly conveying attitude ("My symptoms don't seem to improve with the medication") |
| **Judgement** | Direct attitude toward self ("I'm not good at medication adherence") or others ("The nurses weren't helpful") | Indirectly conveying attitude ("I'm not very consistent about taking my medication") |

### Clinician Role: EO Elicitations

Elicitations are parts of the empathic interaction where the clinician elicits the patient's feelings and perspectives.

| Type | Direct | Indirect |
|------|--------|----------|
| **Feelings** | "How do you feel about that?", "What was your reaction?" | "So you're worried that the treatment won't work" |
| **Appreciation** | "How was the MRI for you?", "Do you find the medication helpful?" | "It sounds like the medication isn't helping" |
| **Judgement** | "Do you think you are a good father?", "Was the nurse helpful?" | "So the nurse was not very helpful then?" |

### Clinician Role: Empathic Responses

#### 1. Acceptance - Unconditional Positive Regard

| Subcategory | Description | Example |
|-------------|-------------|---------|
| Explicit Positive Judgement | Expression of positive judgment of the patient as a person | "You are a reasonable parent", "You're very responsible" |
| Implicit Positive Judgement | Expression of judgement of the patient's thoughts or feelings | "It's really great that you've been taking care of your parents" |
| Repetition | Repeating or paraphrasing patient's words without countering | "I understand you're worried about the cancer spreading" |
| Allowing Full Expression | Allowing patients to express feelings fully through minimal responses | Nodding, avoidance of interruption |

#### 2. Acceptance - Neutral Support

| Subcategory | Description | Example |
|-------------|-------------|---------|
| Explicit Appreciation | Appreciation of ideas, feelings, or behaviors regarding normality | "It's completely normal to be upset about this" |
| Explicit Judgement | Denying negative self-assessment by the patient | "You're not crazy for being worried about that" |

#### 3. Sharing

Sharing patient views or feelings through expressed agreement:
- "I'm sure I would also feel anxious in this situation"
- "Yes, this medication really does not taste good"
- "Oh, no!"

#### 4. Understanding

Understanding and acknowledgement of the patient's views and feelings:
- "I get the sense that you found the other doctor unhelpful"
- "I see that the medication isn't working for you"
- "I understand that you're worried about infections"

---

## Interactional Sequences

The framework identifies three types of interactional sequences through span-relations:

### 1. Empathic Response Sequences
EO is directly linked to an empathic response that explicitly recognizes the attitudes conveyed in the EO.

### 2. EO Continuer Sequences
A "potential EO continuer" facilitates further exploration, leading to more explicit empathic opportunities and increased empathic understanding.
- Sequence: Implicit EO → Elicitation → Explicit EO

### 3. EO Terminating Sequences
Clinician directs the conversation away from an explicit or implicit EO (missed opportunities).

---

## Dataset: BBN Empathy Dataset

### Characteristics
- **Language:** German
- **Conversations:** 63 simulated BBN conversations
- **Participants:** Medical students (as clinicians) and trained standardized patient actors
- **Scenarios:** Cancer diagnoses, treatment failures, informing family members of accidents, etc.
- **Annotation Type:** Span-relation task

### Key Statistics

#### Empathic Opportunities Distribution
| Type | Explicit (Avg) | Implicit (Avg) |
|------|---------------|----------------|
| Feelings | 2.4 ± 2.3 | 12.5 ± 5.6 |
| Appreciation | 5.9 ± 5.0 | 2.4 ± 2.6 |
| Judgement | 9.5 ± 6.0 | 1.2 ± 1.7 |

**Key Finding:** Implicit feelings are much more common than explicit feelings, consistent with prior research by Suchman et al. (1997).

#### EO Response Rates
| EO Type | No Response | Response |
|---------|-------------|----------|
| Explicit Feeling | 35.4% | 64.6% |
| Implicit Feeling | 27.8% | 72.2% |
| Explicit Appreciation | 46.9% | 53.1% |
| Implicit Appreciation | 36.5% | 63.5% |
| Explicit Judgement | 34.3% | 65.7% |
| Implicit Judgement | 27.3% | 72.7% |

**Key Finding:** Implicit EOs have a higher rate of clinician responses than explicit EOs.

---

## Applications

The BBN Empathy Framework enables:

1. **NLP Model Training** - Automatic labeling of empathic opportunities, elicitations, and responses
2. **Medical Education Tools** - Automated feedback systems for communication skills training
3. **Empathy Detection** - Identifying presence and subtypes of empathic behaviors
4. **Empathy Generation** - Informing models that generate empathetic responses
5. **Research** - Understanding causal relations in empathic exchanges

---

## Key Contributions

1. Novel annotation framework for capturing discourse elements and relations of empathic interactions in clinical scenarios
2. Integration with SPIKES protocol for structural stages of BBN conversations
3. First dataset containing discourse labels and relations for clinical empathy
4. Contribution to empathy language resources in non-English languages (German)

---

## References

- Pounds, G. (2011). Appraisal framework for clinical empathy
- Suchman, A. L., Markakis, K., Beckman, H. B., & Frankel, R. (1997). Model of empathic communication in medical conversations
- Baile, W. F., et al. (2000). SPIKES protocol for breaking bad news
- Martin, J. R., & White, P. R. R. (2005). Appraisal framework (Systemic Functional Linguistics)
