"""
assemble_dataset.py
Builds train.jsonl / eval.jsonl for the NIH-reviewer voice-transfer fine-tune.

The task: rewrite a generic critique paragraph into the voice of an NIH study
section reviewer. NIH reviewer voice has recognizable conventions:
  - Strengths and Weaknesses are explicit, often section-headed.
  - Concerns cite rigor, reproducibility, statistical power, prior data.
  - Language is hedged but pointed ("The PI does not address...", "It is not
    clear how...").
  - Recommendations are framed as questions or as missing analyses, not as
    directives.

The sample below produces ~15 pairs. For a real fine-tune, replace
SEED_PAIRS with 80-150 entries authored by an experienced NIH reviewer.

By Juan B. Gutierrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import json
import os
import random


SYSTEM_PROMPT = (
    "Rewrite the input critique paragraph in the voice of an experienced NIH "
    "study section reviewer. Use section-style language (Strengths / Weaknesses "
    "if appropriate). Reference rigor, reproducibility, statistical justification, "
    "feasibility, and prior data where relevant. Pose concerns as questions or as "
    "missing analyses rather than directives. Maintain a measured, hedged tone. "
    "Output only the rewritten critique."
)


# (generic_critique, nih_voice_critique)
SEED_PAIRS = [
    (
        "The study sample is too small to be reliable.",
        "The proposed sample size raises concerns about statistical power. A formal power calculation is not provided, and given the effect size assumed in Aim 2, the planned N may be insufficient to detect the hypothesized between-group differences. The PI should provide a power calculation with explicit assumptions about variance and dropout."
    ),
    (
        "The methods section is unclear about how data will be analyzed.",
        "The analytic plan is underspecified. The PI states that mixed-effects models will be used but does not describe the random-effects structure, model selection strategy, or approach to multiple comparisons. Without these details, it is difficult to evaluate rigor and reproducibility."
    ),
    (
        "The researchers should explain how they will recruit minority participants.",
        "Recruitment of a diverse cohort is a stated goal, but the application does not describe concrete strategies for engaging underrepresented populations. Given the demographic composition of the catchment area, it is not clear whether the proposed recruitment timeline is feasible. A more developed plan, ideally with letters from community partners, would strengthen the Approach."
    ),
    (
        "The preliminary data is weak.",
        "The preliminary data, while suggestive, do not yet establish feasibility for the proposed mechanistic studies. Figure 2 shows a trend in the predicted direction, but the small N and absence of error bars make it difficult to assess robustness. Additional pilot data would substantially strengthen the Significance and Approach."
    ),
    (
        "The PI seems qualified for the work.",
        "The PI's track record demonstrates productive work in adjacent areas and the assembled team is well qualified for the proposed studies. The senior personnel collectively cover the necessary methodological domains, and the institutional environment is appropriate."
    ),
    (
        "The innovation here is questionable; this has been done before.",
        "The Innovation, as framed, is modest. The proposed approach extends prior work by the PI's own group (Smith et al., 2022) and by others in the field. The application would benefit from a clearer articulation of what is genuinely novel about the proposed methods or conceptual framework relative to existing literature."
    ),
    (
        "The hypothesis is not specific enough.",
        "The central hypothesis, as stated in Aim 1, is broad and does not generate falsifiable predictions. A reviewer would be better positioned to evaluate the proposed experiments if the hypothesis specified expected directionality, magnitude, or mechanism. As written, the alternative hypothesis cannot be clearly distinguished from the null."
    ),
    (
        "The budget seems too high for what they want to do.",
        "Budgetary justification appears generous in several categories. Equipment costs in Year 1 are not accompanied by a clear rationale for institutional cost-sharing alternatives, and the personnel effort allocations exceed what is typical for the proposed scope of work. A more parsimonious budget would not compromise the science."
    ),
    (
        "I'm worried they can't actually do this in 5 years.",
        "Feasibility within the proposed five-year timeline is a significant concern. Aim 3 requires recruitment, follow-up, and analytic completion of a longitudinal cohort that historically has required longer windows in this PI's prior work. The application does not adequately address how this timeline will be achieved."
    ),
    (
        "The animal model isn't a good match for the human disease.",
        "Translational relevance of the proposed animal model is a notable weakness. The chosen model recapitulates some but not all features of the human disease; in particular, it lacks the inflammatory phenotype central to the proposed mechanistic hypothesis. The PI should justify model choice or consider parallel work in a complementary model."
    ),
    (
        "The data sharing plan looks fine.",
        "The Data Sharing Plan is consistent with NIH policy and identifies an appropriate public repository with a reasonable timeline. Metadata standards are specified and FAIR principles are addressed."
    ),
    (
        "They don't address sex differences.",
        "Consideration of sex as a biological variable is not adequately addressed. The proposed analyses do not stratify by sex, and the rationale for this choice is not provided. NOT-OD-15-102 anticipates analytic plans that account for sex; the application should either pre-specify sex-stratified analyses or justify their absence."
    ),
    (
        "The collaborators are good but it's not clear what they're actually doing.",
        "The assembled collaborators bring relevant expertise; however, the application does not specify the percent effort or the discrete deliverables each will contribute. Letters of support are general rather than tied to specific aims, making it difficult to evaluate the operational reality of the proposed collaborations."
    ),
    (
        "The statistical methods are reasonable but I'd like to see how they handle missing data.",
        "Proposed analyses are appropriate to the design, but the application does not describe the strategy for handling missing data. Given the longitudinal design and the anticipated attrition rate, sensitivity analyses under different missingness mechanisms (MCAR, MAR) would strengthen confidence in the primary findings."
    ),
    (
        "The biosketch looks productive but I don't see much recent first-author work in this area.",
        "The PI's biosketch documents productive collaborative work; however, recent first- or senior-author publications in the specific area of the proposed studies are sparse. This raises a modest concern about the PI's independent footprint in this domain, partially mitigated by the strength of the assembled team."
    ),
]


def make_example(generic, nih_voice):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": generic},
            {"role": "assistant", "content": nih_voice},
        ]
    }


def main(seed=7, eval_size=4):
    random.seed(seed)
    pairs = list(SEED_PAIRS)
    random.shuffle(pairs)

    eval_pairs = pairs[:eval_size]
    train_pairs = pairs[eval_size:]

    here = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(here, "train.jsonl")
    eval_path = os.path.join(here, "eval.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for g, n in train_pairs:
            f.write(json.dumps(make_example(g, n)) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for g, n in eval_pairs:
            f.write(json.dumps(make_example(g, n)) + "\n")

    print(f"Wrote {len(train_pairs)} training examples to {train_path}")
    print(f"Wrote {len(eval_pairs)} eval examples to {eval_path}")


if __name__ == "__main__":
    main()
