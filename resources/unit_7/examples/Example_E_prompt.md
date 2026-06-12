# Example_E — Block C demo prompt

This file is the prompt fed to the LLM ensemble during the Unit 7.2 Block C
instructor-led demonstration. The class watches the same prompt run twice:
**cold** (no context) and **grounded** (sociodemographic coding history
injected as a system prompt). The two outputs are compared.

---

## Task (single-paragraph form, for the ensemble)

> Study the data dictionary at `data/Nat1969-71doc.pdf` and the fixed-width
> flat file at `data/US1969.dat`. Build a single Python program that:
> (a) extracts **mortality** and **maternal level of education** by **race**
> from the data, aggregated **by state**;
> (b) writes the results as tidy CSV files; and
> (c) visualizes the by-state results as a US map (or, if a map renderer
> is not available, as a state-by-variable heatmap).
>
> Place all artifacts in `resources/unit_7/examples/` with the prefix
> `Example_E_` so they cluster with the rest of the unit's runnable
> examples.

---

## Inputs

| File | Path | Purpose |
|---|---|---|
| Data dictionary | `data/Nat1969-71doc.pdf` | NCHS DDP-PB record layout, 1969–1971 |
| Flat file | `data/US1969.dat` | Fixed-width, 215 bytes per record, ~3.5M rows |

---

## Cold-run constraints (first pass — no grounding)

- The LLM is given the prompt above and the dictionary.
- The LLM is not warned about NCHS race coding history, period-specific
  category labels, or the absence of an in-record infant mortality
  variable.
- Class observes what the LLM proposes: does it conflate "mortality"
  with prior-fetal-deaths-to-mother? Does it preserve period terminology
  literally? Does it state assumptions, or paper over them?

## Grounded-run context (second pass — system prompt injected)

The instructor injects the following as a system prompt before the
task above is re-run:

> **Context for this query.** The NCHS 1969–1971 natality file at
> `data/US1969.dat` contains **birth records only**. It does not contain
> infant mortality at the record level. The closest in-record mortality
> *risk* indicator is birthweight (positions 73–76; recode at 79). The
> two prior-history fields at positions 54–55 and 56–57 record the
> mother's *prior* live births now dead and prior fetal deaths, not
> this birth's outcome.
>
> **Race coding (positions 37–40).** The detail race codes use 1969-era
> terminology including the term *Negro*. The three-category recode at
> position 40 collapses to `{1=White, 2=All Other Excluding Negro,
> 3=Negro}`. Hispanic ethnicity is *not separately coded* in this file.
> The race assignment rules were revised after 1977; any analysis
> spanning 1969–1986 must treat race as a temporally unstable variable.
>
> **Maternal education (positions 98–102).** Detail (98–99) uses
> 00–08 for elementary years, 09–17 for high school + college, 66 for
> unknown, 77 for "cannot classify", 88 for non-reporting states, 99
> for no entry. Recode 14 (100–101) and Recode 6 (102) compress these
> categories differently. Pick one and document the choice.
>
> **Place of residence (positions 13–14).** State codes 01–51 are the
> 50 states + DC in alphabetical order. Codes 52–59 are non-US (1970–71
> only). For 1969, codes 52–59 are absent.
>
> **Reproducibility note.** Document every disambiguation in code
> comments. Log every assumption as an invalidation log entry in the
> governance protocol format.

---

## Comparative question (the heart of the demo)

For each variable (race / education / state of residence), the class
records:

1. Did the ensemble flag the variable as temporally unstable
   **without prompting**?
2. Did the ensemble use period-correct terminology, or sanitized
   modern terms?
3. Did the ensemble surface the "no in-record mortality" issue, or
   silently substitute a proxy?
4. Did the ensemble produce runnable code on the first attempt, or
   require multiple rounds?

The differences between the cold and grounded runs become entries in
each team's invalidation log.

---

## Deliverable from the *grounded* run

A working `Example_E.py` (in this folder) that:

- streams `data/US1969.dat` line by line (do not load 381 MB into memory),
- extracts year, residence-state, race-recode-3, education-recode-6,
  and birthweight-recode-3,
- aggregates counts by (state × race) and (state × education),
- writes `Example_E_state_race.csv`,
       `Example_E_state_education.csv`, and
       `Example_E_state_race_lowbirthweight.csv` (the mortality proxy),
- renders heatmap PNGs as a robust fallback when no map renderer is
  available,
- prints, at the end, the count for the Block B reference question:
  *live births in 1969 to mothers residing in Texas*, with the
  exact filter used (so the disambiguation decisions made by the
  ensemble are auditable).

The file `Example_E.py` in this folder is the version produced by the
**grounded** run. Compare it to whatever the cold run produces in
your live session.
