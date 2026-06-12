"""
Example_E.py
Block C instructor-led demo for Unit 7.2.

Question:
    What is the MEDIAN years of education for White mothers and Black
    mothers in the 1969 NCHS natality file, nationally and by state of
    residence?

The script is the output of the GROUNDED LLM-ensemble run described in
Example_E_prompt.md (with the sociodemographic context injected). Every
disambiguation decision is named in code.

KEY DESIGN DECISIONS
--------------------

1.  Race field: position 38 (Detail Race of Mother). The NCHS dictionary
    has NO "Mother Race Recode 3" -- only Detail. 1969-era codes:
        1 = White
        2 = Negro    (period term used in the 1969 NCHS dictionary)
        3-9 = other categories, excluded from this White/Black comparison
    We surface code 2 with the period label, flagged as such.

2.  Education field: position 98-99 (Detail). Codes 00-17 map directly
    to years of education (00 = no schooling .. 17 = 5+ years college).
    We use Detail because it preserves resolution; Recode 14 collapses
    0-5 years, Recode 6 is too coarse for a median.

3.  Right-censoring at code 17. "5+ years college" is right-censored.
    We use MEDIAN (not mean) precisely because it is censoring-robust
    whenever the median falls strictly below 17. If a cell's median
    equals 17, we flag it as censored in the output.

4.  Exclusion rules (record-level):
        66 = Unknown          -> excluded
        77 = Cannot classify  -> excluded
        88 = Non-reporting    -> excluded (state of OCCURRENCE)
        99 = No entry         -> excluded
    The exclusion count by reason is reported per (state, race) cell.
    Note: 88 is keyed on the OCCURRENCE state, not residence. So a
    Texas-resident mother who gave birth in a non-reporting state
    shows up as 88 in the Texas-residence cell.

5.  State grouping: residence state (positions 13-14), consistent with
    Block B's Texas-residence question.

By Juan B. Gutierrez, Professor of Mathematics
University of Texas at San Antonio.

License: Creative Commons Attribution-ShareAlike 4.0 International
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------
# Fixed-width field map (1-indexed positions, NCHS DDP-PB layout 1969-71)
# ---------------------------------------------------------------------
FIELDS = {
    'year':              (1, 1),       # 9=1969, 0=1970, 1=1971
    'res_state':         (13, 14),     # 01-51 alphabetical (50 states + DC)
    'mother_race':       (38, 38),     # detail race of mother (1=White, 2=Negro, ...)
    'mother_edu_detail': (98, 99),     # 00-17 = years; 66/77/88/99 = missing
}

# NCHS alphabetical state codes (01..51), with DC alphabetical (between DE and FL).
STATE_NAMES = [
    None,
    "Alabama", "Alaska", "Arizona", "Arkansas", "California",
    "Colorado", "Connecticut", "Delaware", "District of Columbia", "Florida",
    "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana",
    "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
    "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin",
    "Wyoming",
]
STATE_ABBR = [
    None,
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI",
    "WY",
]

# Tile-map grid (row, col), rough US geographic correspondence.
# 8 rows x 11 cols. Used by both the per-race state maps.
TILE_GRID = {
    "ME": (0, 10),
    "VT": (1, 9),  "NH": (1, 10),
    "WA": (2, 0),  "MT": (2, 2), "ND": (2, 3), "MN": (2, 4), "WI": (2, 5),
    "MI": (2, 7),  "NY": (2, 8), "MA": (2, 10),
    "OR": (3, 0),  "ID": (3, 1), "WY": (3, 2), "SD": (3, 3), "IA": (3, 4),
    "IL": (3, 5),  "IN": (3, 6), "OH": (3, 7), "PA": (3, 8), "NJ": (3, 9),
    "CT": (3, 10),
    "CA": (4, 0),  "NV": (4, 1), "UT": (4, 2), "CO": (4, 3), "NE": (4, 4),
    "MO": (4, 5),  "KY": (4, 6), "WV": (4, 7), "VA": (4, 8), "MD": (4, 9),
    "RI": (4, 10),
                   "AZ": (5, 1), "NM": (5, 2), "KS": (5, 3), "AR": (5, 4),
    "TN": (5, 5),  "NC": (5, 6), "SC": (5, 7), "DC": (5, 8), "DE": (5, 9),
    "HI": (6, 0),                              "OK": (6, 3), "LA": (6, 4),
    "MS": (6, 5),  "AL": (6, 6), "GA": (6, 7),
    "AK": (7, 0),                              "TX": (7, 3),
                                                              "FL": (7, 7),
}
GRID_ROWS, GRID_COLS = 8, 11

RACE_CODES_KEEP = {'1': 'White', '2': 'Black (NCHS "Negro")'}
EDU_VALID = set(f"{i:02d}" for i in range(0, 18))   # '00'..'17'
EDU_MISSING_REASONS = {'66': 'unknown', '77': 'unclassifiable',
                       '88': 'non-reporting state', '99': 'no entry'}

# Recode-6 buckets, defined by which Detail codes (00-17) fall into each.
# Used by the stacked-distribution chart.
EDU_BUCKETS = [
    ('0-8 years',     list(range(0, 9))),
    ('9-11 years',    list(range(9, 12))),
    ('12 years (HS)', [12]),
    ('13-15 years',   list(range(13, 16))),
    ('16+ years',     [16, 17]),
]


def slice_field(record: str, name: str) -> str:
    s, e = FIELDS[name]
    return record[s - 1 : e]


def state_idx(code: str) -> int | None:
    if not code or not code.isdigit():
        return None
    n = int(code)
    if 1 <= n <= 51:
        return n
    return None


# ---------------------------------------------------------------------
# One streaming pass: build histograms of education code per (state, race).
# Also tally exclusions by reason.
# ---------------------------------------------------------------------
def process(data_path: Path, year_filter: str = '9') -> dict:
    # hist[(state_code, race_code)][edu_code_int] = count
    hist = defaultdict(lambda: defaultdict(int))
    # excl[(state_code, race_code)][reason] = count
    excl = defaultdict(lambda: defaultdict(int))
    total = 0
    skipped = 0

    expected_len = 215
    with open(data_path, 'r', encoding='latin-1', errors='replace') as fh:
        for line in fh:
            rec = line.rstrip('\n').rstrip('\r')
            if len(rec) < expected_len:
                skipped += 1
                continue
            year = slice_field(rec, 'year')
            if year_filter and year != year_filter:
                continue
            total += 1

            race = slice_field(rec, 'mother_race')
            if race not in RACE_CODES_KEEP:
                continue

            res = slice_field(rec, 'res_state')
            edu = slice_field(rec, 'mother_edu_detail')

            if edu in EDU_VALID:
                hist[(res, race)][int(edu)] += 1
            else:
                reason = EDU_MISSING_REASONS.get(edu, f'other({edu})')
                excl[(res, race)][reason] += 1

    return {'hist': hist, 'excl': excl, 'total': total, 'skipped': skipped}


# ---------------------------------------------------------------------
# Grouped-data median with linear interpolation.
# Treats integer code k as the midpoint of bin [k-0.5, k+0.5] and
# interpolates within the bin where the cumulative crosses n/2. This
# kills the discretization artifact that makes the histogram-based
# median jump by full units between groups whose distributions are
# nearly identical at the central bin boundary.
#
# Formula (standard grouped-data median):
#     median = L + ((n/2 - F) / f) * h
# where L = lower bin edge, F = cumulative count before the median
# bin, f = count in the median bin, h = bin width (= 1 here).
# ---------------------------------------------------------------------
def median_from_hist(h: dict[int, int]) -> tuple[float | None, int, bool]:
    """
    Returns (median_years, n_used, censored_flag).

    censored_flag is True iff the interpolated median falls within the
    right-censored bin centered at 17 (i.e., median >= 16.5).
    """
    n = sum(h.values())
    if n == 0:
        return None, 0, False
    target = n / 2.0
    prev_cum = 0
    cum = 0
    for code in sorted(h.keys()):
        f = h[code]
        prev_cum = cum
        cum += f
        if cum >= target:
            if f == 0:
                median = float(code)
            else:
                median = (code - 0.5) + (target - prev_cum) / f
            return median, n, (median >= 16.5)
    return None, n, False


# ---------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------
def write_summary_csv(out_path: Path, summary_rows: list[dict]) -> None:
    cols = ['group', 'median_years_education', 'n_used',
            'n_excluded', 'censored_at_17']
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)


def write_by_state_csv(out_path: Path, rows: list[dict]) -> None:
    cols = ['state_code', 'state_name', 'state_abbr',
            'white_median', 'white_n', 'white_excluded',
            'black_median', 'black_n', 'black_excluded']
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------
def fmt_med(m):
    return "--" if m is None else f"{m:.1f}"


def render_distribution(hist: dict, summary_rows: list[dict],
                        out_path: Path) -> None:
    """
    Stacked horizontal bar chart of the maternal-education distribution,
    aggregated across all states, with one bar per race. Buckets are the
    Recode-6 categories. Each bar sums to 100% so the shape of the
    distribution -- not the raw sample size -- is what's compared.
    """
    races = [('1', 'White mothers'),
             ('2', 'Black mothers\n(NCHS "Negro")')]

    # Aggregate detail codes into Recode-6 buckets, per race.
    bucket_counts = {race: [0] * len(EDU_BUCKETS) for race, _ in races}
    for (sc, rc), h in hist.items():
        if rc not in {r for r, _ in races}:
            continue
        for code, count in h.items():
            for i, (_label, codes) in enumerate(EDU_BUCKETS):
                if code in codes:
                    bucket_counts[rc][i] += count
                    break

    fig, ax = plt.subplots(figsize=(11, 4.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(EDU_BUCKETS)))

    summary_by_race = {r['group']: r for r in summary_rows}
    for j, (race_code, race_label) in enumerate(races):
        counts = bucket_counts[race_code]
        total = sum(counts)
        if total == 0:
            continue
        props = [c / total for c in counts]
        left = 0.0
        for i, (bucket_label, _codes) in enumerate(EDU_BUCKETS):
            ax.barh(j, props[i], left=left, height=0.7,
                    color=colors[i], edgecolor='white', linewidth=1.2,
                    label=bucket_label if j == 0 else None)
            if props[i] >= 0.04:
                # Pick text color by luminance
                rgba = colors[i]
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(left + props[i] / 2, j,
                        f"{props[i] * 100:.0f}%",
                        ha='center', va='center', fontsize=10,
                        color='white' if lum < 0.55 else 'black')
            left += props[i]

        # Right-side annotation: interpolated median + n
        plain_label = race_label.replace('\n', ' ').replace('  ', ' ')
        # Match against the summary using the canonical 'White' / 'Black ...' name
        for key in summary_by_race:
            if key.startswith(plain_label.split()[0]):
                med = summary_by_race[key]['median_years_education']
                n = summary_by_race[key]['n_used']
                ann = f"median = {med:.2f}   n = {n:,}" if med is not None \
                      else f"median = --   n = {n:,}"
                ax.text(1.01, j, ann, transform=ax.get_yaxis_transform(),
                        ha='left', va='center', fontsize=9)
                break

    ax.set_yticks(range(len(races)))
    ax.set_yticklabels([label for _, label in races])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Proportion of mothers (record-level)')
    ax.set_title('1969 NCHS Natality: maternal-education distribution by race\n'
                 '(Detail education re-bucketed to Recode-6 categories)',
                 fontsize=11)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=5,
              fontsize=9, frameon=False)
    fig.subplots_adjust(right=0.78)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def render_heatmap(rows: list[dict], out_path: Path) -> None:
    M = np.full((51, 2), np.nan)
    state_lbls = []
    for r in rows:
        i = int(r['state_code']) - 1
        if r['white_median'] is not None:
            M[i, 0] = r['white_median']
        if r['black_median'] is not None:
            M[i, 1] = r['black_median']
        state_lbls.append(f"{r['state_abbr']}  {r['state_name']}")

    fig, ax = plt.subplots(figsize=(7, 12))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='#dddddd')
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=10.5, vmax=12.5)
    ax.set_yticks(range(51))
    ax.set_yticklabels(state_lbls, fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['White mothers', 'Black mothers\n(NCHS "Negro")'],
                       fontsize=9)
    ax.set_title('Median years of education by state of residence',
                 fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.set_label('Median years', fontsize=9)
    for i in range(51):
        for j in range(2):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}",
                        ha='center', va='center', fontsize=6,
                        color='white' if M[i, j] < 11.5 else 'black')
            else:
                ax.text(j, i, '--', ha='center', va='center',
                        fontsize=6, color='#666666')
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def render_tile_map(rows: list[dict], out_path: Path) -> None:
    """
    Three-panel tile map of US states:
      [0] White mothers -- median years of education
      [1] Black mothers -- median years of education
      [2] Combined exclusion rate (records dropped for code 66/77/88/99)

    Panels 0 and 1 share a viridis 10.5-12.5 color scale, tight enough
    to show within-state White-vs-Black gaps. Panel 2 uses an inverted
    magma scale because 'more excluded' is conventionally darker.
    """
    by_abbr = {r['state_abbr']: r for r in rows}

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    cmap_med = plt.cm.viridis
    norm_med = plt.Normalize(vmin=10.5, vmax=12.5)
    cmap_exc = plt.cm.magma_r
    norm_exc = plt.Normalize(vmin=0.0, vmax=0.6)

    panel_specs = [
        ('white',     'White mothers',                cmap_med, norm_med, False),
        ('black',     'Black mothers (NCHS "Negro")', cmap_med, norm_med, False),
        ('exclusion', 'Exclusion rate (both races)',  cmap_exc, norm_exc, True),
    ]

    for ax, (key, title, cmap, norm, is_pct) in zip(axes, panel_specs):
        for abbr, (rr, cc) in TILE_GRID.items():
            row = by_abbr.get(abbr)
            if key == 'exclusion':
                if row:
                    used = row['white_n'] + row['black_n']
                    excl = row['white_excluded'] + row['black_excluded']
                    total = used + excl
                    val = excl / total if total > 0 else None
                else:
                    val = None
            else:
                val = row[f'{key}_median'] if row else None

            x, y = cc, GRID_ROWS - 1 - rr
            if val is not None:
                rgba = cmap(norm(val))
                face = rgba
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = 'white' if lum < 0.5 else 'black'
            else:
                face = '#dddddd'
                txt_color = '#666666'

            rect = mpatches.Rectangle(
                (x - 0.45, y - 0.45), 0.9, 0.9,
                facecolor=face, edgecolor='black', linewidth=0.8)
            ax.add_patch(rect)
            ax.text(x, y + 0.15, abbr, ha='center', va='center',
                    fontsize=9, fontweight='bold', color=txt_color)
            if val is None:
                label = '--'
            elif is_pct:
                label = f"{val * 100:.0f}%"
            else:
                label = f"{val:.1f}"
            ax.text(x, y - 0.18, label, ha='center', va='center',
                    fontsize=8, color=txt_color)
        ax.set_xlim(-1, GRID_COLS)
        ax.set_ylim(-1, GRID_ROWS)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=12)

    # Two colorbars below the row.
    sm_med = plt.cm.ScalarMappable(cmap=cmap_med, norm=norm_med)
    sm_med.set_array([])
    cbar1 = fig.colorbar(sm_med, ax=axes[:2], orientation='horizontal',
                         fraction=0.05, pad=0.02, shrink=0.7)
    cbar1.set_label('Median years of education (interpolated)', fontsize=10)

    sm_exc = plt.cm.ScalarMappable(cmap=cmap_exc, norm=norm_exc)
    sm_exc.set_array([])
    cbar2 = fig.colorbar(sm_exc, ax=axes[2], orientation='horizontal',
                         fraction=0.05, pad=0.02, shrink=0.85)
    cbar2.set_label('Fraction of records excluded', fontsize=10)

    fig.suptitle(
        '1969 NCHS Natality: maternal education by state of residence',
        fontsize=13)
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    p.add_argument('--data', type=Path,
                   default=here.parent.parent.parent / 'data' / 'US1969.dat')
    p.add_argument('--year', type=str, default='9')
    p.add_argument('--outdir', type=Path, default=here)
    args = p.parse_args()

    if not args.data.exists():
        print(f"ERROR: data file not found: {args.data}", file=sys.stderr)
        return 1

    print(f"[Example_E] Reading: {args.data}")
    print(f"[Example_E] Year filter: {args.year!r} (9=1969)")

    res = process(args.data, args.year)
    print(f"[Example_E] Records processed: {res['total']:,}")
    print(f"[Example_E] Records skipped (short lines): {res['skipped']:,}")

    # National summary (sum histograms over all states).
    summary_rows = []
    for race_code in ('1', '2'):
        agg = defaultdict(int)
        excl_agg = defaultdict(int)
        for (sc, rc), h in res['hist'].items():
            if rc != race_code:
                continue
            for k, v in h.items():
                agg[k] += v
        for (sc, rc), e in res['excl'].items():
            if rc != race_code:
                continue
            for k, v in e.items():
                excl_agg[k] += v
        median, n_used, censored = median_from_hist(agg)
        n_excl = sum(excl_agg.values())
        summary_rows.append({
            'group': RACE_CODES_KEEP[race_code],
            'median_years_education': median,
            'n_used': n_used,
            'n_excluded': n_excl,
            'censored_at_17': censored,
        })

    # Per-state.
    state_rows = []
    for code in range(1, 52):
        sc = f"{code:02d}"
        white_h = res['hist'].get((sc, '1'), {})
        black_h = res['hist'].get((sc, '2'), {})
        white_excl = sum(res['excl'].get((sc, '1'), {}).values())
        black_excl = sum(res['excl'].get((sc, '2'), {}).values())
        wmed, wn, _ = median_from_hist(white_h)
        bmed, bn, _ = median_from_hist(black_h)
        state_rows.append({
            'state_code': sc,
            'state_name': STATE_NAMES[code],
            'state_abbr': STATE_ABBR[code],
            'white_median': wmed,
            'white_n': wn,
            'white_excluded': white_excl,
            'black_median': bmed,
            'black_n': bn,
            'black_excluded': black_excl,
        })

    # CSVs
    csv_sum = args.outdir / 'Example_E_education_summary.csv'
    csv_state = args.outdir / 'Example_E_education_by_state.csv'
    write_summary_csv(csv_sum, summary_rows)
    write_by_state_csv(csv_state, state_rows)
    print(f"[Example_E] Wrote {csv_sum.name}, {csv_state.name}")

    # Visualizations
    dist_path = args.outdir / 'Example_E_distribution.png'
    heat_path = args.outdir / 'Example_E_heatmap.png'
    map_path = args.outdir / 'Example_E_map.png'
    render_distribution(res['hist'], summary_rows, dist_path)
    render_heatmap(state_rows, heat_path)
    render_tile_map(state_rows, map_path)
    print(f"[Example_E] Wrote {dist_path.name}, {heat_path.name}, {map_path.name}")

    # Stdout summary
    print()
    print("=" * 70)
    print("National summary (median years of education, by mother's race)")
    print("=" * 70)
    for r in summary_rows:
        cens = " (CENSORED at 17)" if r['censored_at_17'] else ""
        med = "--" if r['median_years_education'] is None \
              else f"{r['median_years_education']:.2f}"
        print(f"  {r['group']:30s}  median = {med:>6s}{cens:18s}  "
              f"n = {r['n_used']:>10,}  (excluded {r['n_excluded']:,})")
    print()

    # Surface the non-reporting / sparse-data residence states explicitly.
    print("Per-state cells with NO usable observations (median = '--'):")
    any_missing = False
    for r in state_rows:
        misses = []
        if r['white_median'] is None and r['white_excluded'] + r['white_n'] > 0:
            misses.append('white')
        if r['black_median'] is None and r['black_excluded'] + r['black_n'] > 0:
            misses.append('black')
        if r['white_n'] + r['white_excluded'] == 0:
            misses.append('white(no records)')
        if r['black_n'] + r['black_excluded'] == 0:
            misses.append('black(no records)')
        if misses:
            any_missing = True
            print(f"  {r['state_abbr']} {r['state_name']:25s} -> "
                  f"missing: {', '.join(misses)}")
    if not any_missing:
        print("  (none)")

    print()
    print("Reminder: code 88 (non-reporting state) is keyed on OCCURRENCE,")
    print("not RESIDENCE. Records grouped here by residence may still be")
    print("excluded because the mother gave birth in a non-reporting state.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
