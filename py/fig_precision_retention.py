#!/usr/bin/env python3
""
# Precision-retention curve

# Demonstrates the core reproducibility argument: tightening the filtering
# stringency monotonically improves reported precision while collapsing the
# retained fraction of localisations. A single threshold-dependent precision
# value conceals this trade; the curve exposes it.


 # - Each localisation event has a photon count N drawn log-normally.
 # - Per-event CRB precision sigma = L / (4 sqrt(N)) for MINFLUX (Balzarotti 2017),
  #  with a small additive instrumental floor.
 # - CFR is anti-correlated with N: low-photon / poorly-converged events have
 #   higher CFR (worse centring) and worse precision, so a CFR cut preferentially
 #   removes the broad-sigma tail.
 # - Sweeping the CFR threshold from loose -> strict yields the retention curve.




import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# Style:

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica:italic",
    "mathtext.bf": "Helvetica:bold",
    "font.size": 8,
    "axes.linewidth": 1.0,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "legend.fontsize": 7.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,   # editable text in PDF
    "ps.fonttype": 42,
    "svg.fonttype": "none",  # editable text in SVG
})


C_PREC = "#1f4e79"   # deep blue  -> precision
C_RET  = "#c1452d"   # brick red  -> retained fraction
C_POOL = "#9aa7b1"   # grey       -> raw pool / reference
C_BAND = "#1f4e79"   # IQR band

rng = np.random.default_rng(7)


# 1. Synthetic localisation population

# The retained population is a mixture of two physically distinct classes that
# CFR filtering is designed to separate:
#   (i)  "clean" single-emitter events: well centred (low CFR), CRB-limited;
#   (ii) "corrupt" events (multi-emitter coincidences, photophysical transients,
#        poor convergence): high CFR and markedly worse effective precision.
# A loose CFR cut admits both classes; a strict cut isolates class (i). This is
# the mechanism by which the SAME raw dataset yields precision values differing
# by a factor of two or more purely through filter selection.
n_clean   = 22_000
n_corrupt = 40_000
n_events  = n_clean + n_corrupt
L = 40.0            # final-iteration probing diameter (nm)
sigma_floor = 0.6   # residual instrumental floor (nm), drift + aberration

# Class (i): clean single-emitter events

logN_c = rng.normal(loc=np.log(70.0), scale=0.45, size=n_clean)
N_c = np.clip(np.exp(logN_c), 12, None)
sig_c = np.sqrt((L / (4.0 * np.sqrt(N_c)))**2 + sigma_floor**2)
sig_c *= rng.lognormal(0.0, 0.10, n_clean)
# low CFR, tightly concentrated near the centred-zero ideal
cfr_c = np.clip(rng.normal(0.30, 0.13, n_clean), 0.0, 1.2)

# Class (ii): corrupt events

# Lower effective photon yield in the relevant channel + estimator bias inflate
# sigma by a large, broadly distributed factor; CFR is high by construction.
logN_x = rng.normal(loc=np.log(22.0), scale=0.55, size=n_corrupt)
N_x = np.clip(np.exp(logN_x), 8, None)
sig_x = np.sqrt((L / (4.0 * np.sqrt(N_x)))**2 + sigma_floor**2)
sig_x *= rng.lognormal(0.75, 0.32, n_corrupt)   # systematic broadening
# high CFR, spread across the upper range
cfr_x = np.clip(rng.normal(0.85, 0.18, n_corrupt), 0.0, 1.2)

# Combine

sigma = np.concatenate([sig_c, sig_x])
cfr   = np.concatenate([cfr_c, cfr_x])
order = rng.permutation(n_events)
sigma = sigma[order]
cfr   = cfr[order]


# 2. Sweep CFR threshold -> retention curve

cfr_thresholds = np.linspace(1.1, 0.30, 80)   # loose -> strict

median_sigma = np.full_like(cfr_thresholds, np.nan)
q25_sigma    = np.full_like(cfr_thresholds, np.nan)
q75_sigma    = np.full_like(cfr_thresholds, np.nan)
retained     = np.full_like(cfr_thresholds, np.nan)

for i, thr in enumerate(cfr_thresholds):
    keep = cfr <= thr
    frac = keep.mean()
    retained[i] = frac * 100.0
    if keep.sum() > 50:
        s = sigma[keep]
        median_sigma[i] = np.median(s)
        q25_sigma[i]    = np.percentile(s, 25)
        q75_sigma[i]    = np.percentile(s, 75)

# Quantify the headline: precision change factor across the swept range
sig_loose  = np.nanmax(median_sigma)
sig_strict = np.nanmin(median_sigma)
factor = sig_loose / sig_strict
print(f"Median sigma loose  = {sig_loose:.2f} nm")
print(f"Median sigma strict = {sig_strict:.2f} nm")
print(f"Precision change factor across sweep = {factor:.2f}x")
print(f"Retained fraction at strictest shown = {np.nanmin(retained):.1f}%")

# Mark three illustrative operating points (loose / typical / aggressive)
op_cfr = [1.00, 0.65, 0.40]
op_labels = ["Loose", "Typical", "Aggressive"]
op_points = []
for thr in op_cfr:
    j = int(np.argmin(np.abs(cfr_thresholds - thr)))
    op_points.append((cfr_thresholds[j], median_sigma[j], retained[j]))


# 3. Figure: two panels
#    (a) precision-retention curve (dual y-axis vs CFR threshold)
#    (b) precision distribution at the three operating points

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.25, 1.0]}
)

# Panel (a): retention curve

x = cfr_thresholds

# Precision (left axis)
axA.plot(x, median_sigma, color=C_PREC, lw=1.8, zorder=5,
         label="Median precision")
axA.set_xlabel("CFR threshold (\u2190 stricter)")
axA.set_ylabel("Localisation precision $\\sigma_\\mathrm{loc}$ (nm)",
               color=C_PREC)
axA.tick_params(axis="y", colors=C_PREC)
axA.spines["left"].set_color(C_PREC)
axA.set_xlim(0.30, 1.10)
axA.set_ylim(0, np.nanpercentile(q75_sigma, 97) * 1.05)
axA.xaxis.set_major_locator(MultipleLocator(0.2))
axA.xaxis.set_minor_locator(MultipleLocator(0.1))

# Retained fraction (right axis)
axA2 = axA.twinx()
axA2.plot(x, retained, color=C_RET, lw=1.8, ls="-", zorder=4,
          label="Retained fraction")
axA2.set_ylabel("Retained localisations (%)", color=C_RET)
axA2.tick_params(axis="y", colors=C_RET)
axA2.spines["right"].set_color(C_RET)
axA2.spines["left"].set_color(C_PREC)
axA2.set_ylim(0, 102)

# Operating-point markers
label_dy = {"Loose": 11, "Typical": 11, "Aggressive": 20}
for (thr, sig, ret), lab in zip(op_points, op_labels):
    axA.plot(thr, sig, "o", color=C_PREC, ms=5, mec="white", mew=0.8, zorder=6)
    axA2.plot(thr, ret, "s", color=C_RET, ms=5, mec="white", mew=0.8, zorder=6)
    axA.annotate(lab, (thr, sig), textcoords="offset points",
                 xytext=(0, label_dy[lab]), ha="center", fontsize=7,
                 color="#333333")

# Combined legend
h1, l1 = axA.get_legend_handles_labels()
h2, l2 = axA2.get_legend_handles_labels()
axA.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False,
           bbox_to_anchor=(0.02, 0.98), fontsize=7)

# Panel (b): precision distributions at operating points

op_colours = ["#c1452d", "#1f6fb4", "#2e8b57"]  # red, blue, green
bins = np.linspace(0, np.nanpercentile(sigma, 99), 45)

from scipy.stats import gaussian_kde

x_kde = np.linspace(0, np.nanpercentile(sigma, 99), 400)
for (thr, _, ret), lab, col in zip(op_points, op_labels, op_colours):
    keep = cfr <= thr
    s = sigma[keep]
    med = np.median(s)
    kde = gaussian_kde(s, bw_method=0.15)
    axB.plot(x_kde, kde(x_kde), color=col, lw=1.8,
             label=f"{lab} (CFR\u2264{thr:.2f}): {med:.2f} nm, {ret:.0f}%")
    axB.axvline(med, color=col, lw=1.0, ls="--", alpha=0.8)

axB.set_xlabel("Localisation precision $\\sigma_\\mathrm{loc}$ (nm)")
axB.set_ylabel("Probability density")
axB.set_xlim(0, x_kde[-1])
axB.legend(loc="upper left", frameon=False, fontsize=6.8,
           handlelength=1.2, borderaxespad=0.3,
           bbox_to_anchor=(0.42, 1.0))

# Panel labels
for ax, lab in ((axA, "a"), (axB, "b")):
    ax.text(-0.16, 1.04, lab, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

for ax in (axA, axA2, axB):
    ax.tick_params(which="both", top=False)
    ax.minorticks_on()

axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)

fig.tight_layout(w_pad=2.6)
fig.canvas.draw()  

out_pdf = "/Users/hy20365/Downloads/fig_precision_retention.pdf"
out_svg = "/Users/hy20365/Downloads/fig_precision_retention.svg"
out_png = "/Users/hy20365/Downloads/fig_precision_retention.png"

fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=300)
print(f"Saved: {out_pdf}")
print(f"Saved: {out_svg}")
print(f"Saved: {out_png}")
