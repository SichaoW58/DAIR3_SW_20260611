"""
assemble_dataset.py
Builds train.jsonl / eval.jsonl for the bibliography-reformatter fine-tune.

Each training example is a chat completion in the format the OpenAI
fine-tuning API expects:

    {"messages": [
        {"role": "system",    "content": "<task instructions>"},
        {"role": "user",      "content": "<messy citation>"},
        {"role": "assistant", "content": "<NIH-PHS-398 formatted citation>"}
    ]}

This is the canonical SFT pattern: a fixed system prompt + an input + the
desired output. The model learns to map the input style to the output style.

The sample seed list below produces ~20 pairs. For a real fine-tune, replace
SEED_PAIRS with 100-200 entries assembled from your own bibliography.

By Juan B. Gutierrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
"""
import json
import os
import random


SYSTEM_PROMPT = (
    "Reformat the input citation into NIH PHS 398 reference format. "
    "Use the following conventions: 'Authors. Title. Journal. Year;Volume(Issue):Pages.' "
    "List up to six authors; if more, list the first three followed by 'et al.'. "
    "Journal names are abbreviated per Index Medicus. Output only the reformatted "
    "citation; no commentary, no quotation marks."
)


# (messy_input, nih_phs_398_output)
SEED_PAIRS = [
    (
        "Smith, John A. and Jane Doe. (2023). \"Vaccine uptake among adolescents.\" New England Journal of Medicine, vol. 389, no. 12, pp. 1101-1110.",
        "Smith JA, Doe J. Vaccine uptake among adolescents. N Engl J Med. 2023;389(12):1101-1110."
    ),
    (
        "Rodriguez M, Patel S, Chen L, Williams K, Brown T, Davis J, Miller R. A multi-site trial of CBT for depression. JAMA Psychiatry. Published online 2024.",
        "Rodriguez M, Patel S, Chen L, et al. A multi-site trial of CBT for depression. JAMA Psychiatry. 2024."
    ),
    (
        "Garcia-Lopez, M.; Anderson, P. (2022) Mitochondrial dysfunction in Parkinson's disease: a review. NATURE REVIEWS NEUROSCIENCE 23(4): 245-262",
        "Garcia-Lopez M, Anderson P. Mitochondrial dysfunction in Parkinson's disease: a review. Nat Rev Neurosci. 2022;23(4):245-262."
    ),
    (
        "Lee J et al. (2021). Genome-wide association study of asthma in Korean populations. \\textit{American Journal of Human Genetics}, 108(7), 1259--1271.",
        "Lee J, et al. Genome-wide association study of asthma in Korean populations. Am J Hum Genet. 2021;108(7):1259-1271."
    ),
    (
        "Thompson, R. J., O'Brien, M., & Walker, S. (2020). Long-term outcomes of bariatric surgery: a 10-year cohort study. The Lancet 396(10256), 1135-1144.",
        "Thompson RJ, O'Brien M, Walker S. Long-term outcomes of bariatric surgery: a 10-year cohort study. Lancet. 2020;396(10256):1135-1144."
    ),
    (
        "Wang, X.; Liu, Y.; Zhang, H.; Sun, Q. \"Single-cell RNA sequencing reveals heterogeneity in pancreatic ductal adenocarcinoma.\" Cell Reports, Volume 35, Issue 3, 2021, Article 109013.",
        "Wang X, Liu Y, Zhang H, Sun Q. Single-cell RNA sequencing reveals heterogeneity in pancreatic ductal adenocarcinoma. Cell Rep. 2021;35(3):109013."
    ),
    (
        "Davies P. and Kumar S. (2019). 'Microbiome-host interactions in inflammatory bowel disease.' Gastroenterology. 156. 1639-1652.",
        "Davies P, Kumar S. Microbiome-host interactions in inflammatory bowel disease. Gastroenterology. 2019;156:1639-1652."
    ),
    (
        "Nguyen, T. H.; Cooper, R. A.; Foster, B. K. (2024) A randomized controlled trial of mindfulness for chronic pain. Pain Medicine.",
        "Nguyen TH, Cooper RA, Foster BK. A randomized controlled trial of mindfulness for chronic pain. Pain Med. 2024."
    ),
    (
        "Bhatt DL, Steg PG, Mehta SR, Leiter LA, Simon T, Fox K, Held C, Andersson M, Himmelmann A, Ridderstrale W. Ticagrelor in patients with stable coronary disease and diabetes. NEJM 2019 381(14):1309-1320",
        "Bhatt DL, Steg PG, Mehta SR, et al. Ticagrelor in patients with stable coronary disease and diabetes. N Engl J Med. 2019;381(14):1309-1320."
    ),
    (
        "Hernandez, A. F., & Solomon, S. D. (2022). Heart failure with preserved ejection fraction: a review. Journal of the American Medical Association 327(9):871-882",
        "Hernandez AF, Solomon SD. Heart failure with preserved ejection fraction: a review. JAMA. 2022;327(9):871-882."
    ),
    (
        "Kim, J., Park, S., Lee, M. (2023) \"Artificial intelligence for radiology workflow optimization.\" Radiology 308(2):e221234.",
        "Kim J, Park S, Lee M. Artificial intelligence for radiology workflow optimization. Radiology. 2023;308(2):e221234."
    ),
    (
        "Cohen, R.; Stein, M.; Goldberg, D. The role of glymphatic clearance in Alzheimer's pathology. Science Translational Medicine, 2024, 16(729), eabh1234.",
        "Cohen R, Stein M, Goldberg D. The role of glymphatic clearance in Alzheimer's pathology. Sci Transl Med. 2024;16(729):eabh1234."
    ),
    (
        "Patel R, Singh A. (2020). Telemedicine adoption during COVID-19: a systematic review. JMIR; 22(8): e19996.",
        "Patel R, Singh A. Telemedicine adoption during COVID-19: a systematic review. J Med Internet Res. 2020;22(8):e19996."
    ),
    (
        "Mart\\'inez-Ruiz, E. & Sokolov, V. (2021) CRISPR-Cas9 in vivo editing of liver cells. Nature Biotechnology, vol 39, pp 1234-1241",
        "Martinez-Ruiz E, Sokolov V. CRISPR-Cas9 in vivo editing of liver cells. Nat Biotechnol. 2021;39:1234-1241."
    ),
    (
        "Yamamoto T, Suzuki H, Tanaka K, Watanabe M. Long-term safety of immune checkpoint inhibitors in elderly patients. JOURNAL OF CLINICAL ONCOLOGY 41(15): 2750-2761 (2023)",
        "Yamamoto T, Suzuki H, Tanaka K, Watanabe M. Long-term safety of immune checkpoint inhibitors in elderly patients. J Clin Oncol. 2023;41(15):2750-2761."
    ),
    (
        "Wright JM, Musini VM. Cochrane review: First-line drugs for hypertension. Cochrane Database of Systematic Reviews. 2018; (4):CD001841.",
        "Wright JM, Musini VM. First-line drugs for hypertension. Cochrane Database Syst Rev. 2018;(4):CD001841."
    ),
    (
        "Anderson, B. J.; Chen, W. T. (2022). \"A practical guide to single-cell ATAC-seq data analysis.\" Nature Methods 19(5): 545-558.",
        "Anderson BJ, Chen WT. A practical guide to single-cell ATAC-seq data analysis. Nat Methods. 2022;19(5):545-558."
    ),
    (
        "Becker, S., O'Neill, J., and Vargas-Cobas, R. (2020) Air pollution and pediatric asthma exacerbations in Mexico City. Environmental Health Perspectives 128(11):117004",
        "Becker S, O'Neill J, Vargas-Cobas R. Air pollution and pediatric asthma exacerbations in Mexico City. Environ Health Perspect. 2020;128(11):117004."
    ),
    (
        "Foster L; Mehta V; Greene P; Roy A. Statin therapy and incident dementia: a cohort study. ANNALS OF INTERNAL MEDICINE Vol. 174 No. 8 (Apr 2021) 1085--1094.",
        "Foster L, Mehta V, Greene P, Roy A. Statin therapy and incident dementia: a cohort study. Ann Intern Med. 2021;174(8):1085-1094."
    ),
    (
        "Zhou, Q. & Adesina, O. (2024). Machine learning models for sepsis prediction: external validation in three U.S. hospitals. The Lancet Digital Health 6(3): e164--e173.",
        "Zhou Q, Adesina O. Machine learning models for sepsis prediction: external validation in three U.S. hospitals. Lancet Digit Health. 2024;6(3):e164-e173."
    ),
]


def make_example(messy, clean):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": messy},
            {"role": "assistant", "content": clean},
        ]
    }


def main(seed=7, eval_size=5):
    random.seed(seed)
    pairs = list(SEED_PAIRS)
    random.shuffle(pairs)

    eval_pairs = pairs[:eval_size]
    train_pairs = pairs[eval_size:]

    here = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(here, "train.jsonl")
    eval_path = os.path.join(here, "eval.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for messy, clean in train_pairs:
            f.write(json.dumps(make_example(messy, clean)) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for messy, clean in eval_pairs:
            f.write(json.dumps(make_example(messy, clean)) + "\n")

    print(f"Wrote {len(train_pairs)} training examples to {train_path}")
    print(f"Wrote {len(eval_pairs)} eval examples to {eval_path}")


if __name__ == "__main__":
    main()
