"""
master_figures.py
Produces all Python/matplotlib figures for:
  Doughty et al.

Figures produced:
  fig2  - beam quality diagnostic: real vs theoretical LG01 beam
           panels a-c (real) + d-f (theoretical) → donut_comparison.pdf/png
           NOTE: requires xy_nanoparticle_alignment_donut.tif in INPUT dir.
  fig3  — centre-pull bias schematic (panel a) +
           SBR degradation curve (panel b)       → fig3combined.pdf/png
  fig4  — photon budget panel a + PSF/MINFLUX grid panel b
                                                 → figure4.pdf/png
  fig5  — linkage error distributions (panel b only)
                                                 → fig5b_distributions.pdf/png


"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import skewnorm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.lines import Line2D
import tifffile

rng = np.random.default_rng(42)

OUT   = "./output/"
INPUT = "./input/"

import os
os.makedirs(OUT, exist_ok=True)



plt.rcParams.update({
    "font.family":       "Helvetica",
    "font.size":         8,
    "axes.labelsize":    9,
    "axes.linewidth":    1.0,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth":   1.8,
    "legend.fontsize":   7.5,
    "mathtext.fontset":  "custom",
    "mathtext.rm":       "Helvetica",
    "mathtext.it":       "Helvetica:italic",
    "mathtext.bf":       "Helvetica:bold",
})



BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
RED    = "#d62728"


# FIGURE 2a -Centre-pull bias schematic


def hex_positions(cx, cy, r):
    angles = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi, 7)[:-1]
    pts = [(cx + r*np.cos(a), cy + r*np.sin(a)) for a in angles]
    pts.append((cx, cy))
    return pts

def photon_counts_2a(probes, fx, fy, N, b):
    dists2 = np.array([(px-fx)**2 + (py-fy)**2 for px, py in probes])
    L2 = max(dists2[:6])
    return N * dists2 / L2 + b

def naive_estimate(probes, counts):
    op = probes[:6]
    oc = counts[:6]
    inv = 1.0 / (oc + 1e-9)
    wx = sum(c*p[0] for c, p in zip(inv, op)) / sum(inv)
    wy = sum(c*p[1] for c, p in zip(inv, op)) / sum(inv)
    return wx, wy

def draw_geo(ax, r, fx, fy, N, b, title):
    probes = hex_positions(0, 0, r)
    counts = photon_counts_2a(probes, fx, fy, N, b)
    ex, ey = naive_estimate(probes, counts)
    ax.add_patch(Circle((0, 0), r, fill=False, edgecolor="#5b8dd9",
                         lw=1.4, ls="--", zorder=2))
    for px, py in probes[:6]:
        ax.plot(px, py, "o", color="#5b8dd9", ms=6, zorder=5,
                markeredgecolor="white", markeredgewidth=0.7)
    ax.plot(0, 0, "+", color="#5b8dd9", ms=8, mew=1.5, zorder=5)
    Line2D([0],[0], marker="x", color="#bc13fe", markersize=10, mew=2.0,
           linestyle="none", label="True position"),
    Line2D([0],[0], marker="+", color="#27ae60", markersize=10, mew=2.0,
           linestyle="none", label="Estimated position"),ax.plot(fx, fy, "x", color="#bc13fe", ms=10, mew=2.0, zorder=6)
    ax.plot(ex, ey, "+", color="#27ae60", ms=10, mew=2.0, zorder=6)
    if b > 0:
        dx, dy = ex - fx, ey - fy
        dist = np.sqrt(dx**2 + dy**2)
        
        perp_x = -dy/dist * 0.22
        perp_y =  dx/dist * 0.22
        
    ax.set_xlim(-r*1.55, r*1.55)
    ax.set_ylim(-r*1.55, r*1.55)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9, pad=6)
    return probes, counts

def draw_hist(ax, probes, counts, b):
    labels = [f"$x_{i+1}$" for i in range(6)] + ["$x_0$"]
    x   = np.arange(len(probes))
    sig = counts - b
    ax.bar(x, sig, color="#2e86c1", alpha=0.88, zorder=3)
    if b > 0:
        ax.bar(x, [b]*len(probes), bottom=sig,
               color="#e74c3c", alpha=0.80, zorder=3)
        ax.axhline(b, color="#e74c3c", lw=1.0, ls="--", alpha=0.7)
        ax.text(len(probes) - 0.3, b, " $b$", ha="left", va="center",
                fontsize=9, color="#e74c3c")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("EFO (kHz)", fontsize=8)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlabel("Probe position", fontsize=8)

R_2a = 1.0; FX, FY = 0.30, 0.18; N_kHz = 70.0
B_NO = 0.0; B_YES = 30.0

fig2a = plt.figure(figsize=(8.0, 5.2))
fig2a.patch.set_facecolor("white")
gs2a = gridspec.GridSpec(2, 2, height_ratios=[1.45, 1.0],
                          hspace=0.15, wspace=0.32,
                          top=0.93, bottom=0.13)
ax_g1 = fig2a.add_subplot(gs2a[0, 0])
ax_g2 = fig2a.add_subplot(gs2a[0, 1])
ax_h1 = fig2a.add_subplot(gs2a[1, 0])
ax_h2 = fig2a.add_subplot(gs2a[1, 1])

p1, c1 = draw_geo(ax_g1, R_2a, FX, FY, N_kHz, B_NO,  "No background  ($b = 0$)")
p2, c2 = draw_geo(ax_g2, R_2a, FX, FY, N_kHz, B_YES, "With background  ($b > 0$)")
draw_hist(ax_h1, p1, c1, B_NO)
draw_hist(ax_h2, p2, c2, B_YES)

legend_elements = [
    Line2D([0],[0], marker="x", color="#bc13fe", markersize=10, mew=2.0,
           linestyle="none", label="True position"),
    Line2D([0],[0], marker="+", color="#27ae60", markersize=10, mew=2.0,
           linestyle="none", label="Estimated position"),
    Line2D([0],[0], color="#2e86c1", lw=6, alpha=0.88, label="Signal (EFO)"),
    Line2D([0],[0], color="#e74c3c", lw=6, alpha=0.80, label="Background ($b$)"),
]
fig2a.legend(handles=legend_elements, loc="lower center", ncol=4,
             fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

fig2a.savefig(OUT + "fig2a_centrepull.pdf", dpi=300, bbox_inches="tight")
fig2a.savefig(OUT + "fig2a_centrepull.png", dpi=300, bbox_inches="tight")
plt.close(fig2a)
print("✓ fig2a_centrepull")

# FIGURE 2b — SBR precision degradation curve


sbr = np.logspace(-0.5, 2.3, 500)
deg = np.sqrt(1 + 1/sbr)

fig2b, ax = plt.subplots(figsize=(5.5, 4.0))
fig2b.patch.set_facecolor("white")

regimes = [
    (0.5,  2.0,  "#fce4d6", "Tissue\n(depth)"),
    (2.0,  8.0,  "#fef9e7", "Tissue\n(surface)"),
    (8.0,  40.0, "#eaf4fb", "Cultured\ncells"),
    (40.0, 180., "#e9f7ef", "Purified\nprotein"),
]
LABEL_Y = 2.35
for lo, hi, col, lbl in regimes:
    ax.axvspan(lo, hi, color=col, alpha=0.9, zorder=0)
    lx = np.exp((np.log(lo) + np.log(hi)) / 2)
    ax.text(lx, LABEL_Y, lbl, ha="center", va="center",
            fontsize=8.5, color="#333", linespacing=1.5)

ax.plot(sbr, deg, color="#1a5276", lw=2.2, zorder=3)
ax.axhline(1.0, color="#aab", lw=0.9, ls="--", zorder=2)
ax.text(160, 1.015, r"$\sigma_{\rm loc}$ (shot-noise limit)",
        ha="right", va="bottom", fontsize=7, color="#888")

spot_sbrs = [0.9, 4.0, 18.0, 90.0]
spot_cols = ["#c0392b", "#d68910", "#2471a3", "#1e8449"]
for s, c in zip(spot_sbrs, spot_cols):
    ax.plot(s, np.sqrt(1+1/s), "o", color=c, ms=6, zorder=4,
            markeredgecolor="white", markeredgewidth=0.8)

ax.set_xscale("log")
ax.set_xlim(0.5, 180); ax.set_ylim(0.95, 2.55)
ax.set_xlabel("Signal-to-background ratio (SBR)", fontsize=9)
ax.set_ylabel(r"Precision degradation factor  $(1 + 1/\mathrm{SBR})^{1/2}$",
              fontsize=9)
ax.tick_params(labelsize=8)
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout(pad=0.8)
fig2b.savefig(OUT + "fig2b_degradation.pdf", dpi=300, bbox_inches="tight")
fig2b.savefig(OUT + "fig2b_degradation.png", dpi=300, bbox_inches="tight")
plt.close(fig2b)
print("✓ fig2b_degradation")


# FIGURE 3 — Photon budget (panel a) + PSF/MINFLUX grid (panel b)


# Shared physics parameters 
s_nm       = 155.0   # PSF sigma (nm); λ=670nm, NA=1.4 practical
a_px       = 130.0   # pixel size (nm)
L_MF       = 40.0    # MINFLUX final-iteration TCP diameter (nm)
floor_SMLM = 2.0     # nm- residual drift/aberration floor (SMLM)
floor_MF   = 0.5     # nm - active stabilisation floor (MINFLUX)
N_total    = 8000    # AF647 total photon budget (Dempsey 2011, 5 cycles × 1600 ph)
N_bleach   = 1600    # early bleach: 1 switching cycle
b2_vals    = [0, 10, 50]  # background levels ph/px

def sigma_TM(n, b2):
    """Thompson-Mortensen precision with drift/aberration floor."""
    base = (s_nm**2 + a_px**2/12.0) / n
    corr = 1.0 + (4*np.pi*s_nm**2*b2) / (n*a_px**2)
    return np.sqrt(base*corr + floor_SMLM**2)

def sigma_MF(N):
    """MINFLUX CRB per coordinate: σ = L/(4√N), with floor."""
    return np.sqrt((L_MF / (4.0*np.sqrt(N)))**2 + floor_MF**2)

# Panel a: precision vs photons 
N_ax = np.logspace(np.log10(1), np.log10(1e5), 600)
ls_styles = ["-", "--", ":"]
alphas_ls  = [1.0, 0.85, 0.85]

fig3 = plt.figure(figsize=(7.2, 7.2))
gs3  = gridspec.GridSpec(2, 1, figure=fig3,
                          height_ratios=[1.15, 1.55],
                          hspace=0.42,
                          left=0.07, right=0.97,
                          top=0.96, bottom=0.06)
ax3a = fig3.add_subplot(gs3[0])

for b2, ls, alpha in zip(b2_vals, ls_styles, alphas_ls):
    ax3a.loglog(N_ax, sigma_TM(N_ax, b2),
                color=BLUE, lw=1.8, ls=ls, alpha=alpha, zorder=3)
ax3a.loglog(N_ax, sigma_MF(N_ax), color=ORANGE, lw=1.8, zorder=3)

mask_bl     = N_ax <= N_bleach
sigma_at_bl = sigma_TM(np.array([float(N_bleach)]), 0)[0]
sigma_at_tot= sigma_TM(np.array([float(N_total)]),  0)[0]
ax3a.loglog(N_ax[mask_bl], sigma_TM(N_ax, 0)[mask_bl],
            color=RED, lw=1.8, zorder=4)
ax3a.plot(N_bleach, sigma_at_bl, "o", color=RED, ms=6,
          mec="white", mew=1.0, zorder=5)
ax3a.axvline(N_bleach, color=RED,  lw=0.8, ls="--", alpha=0.45, zorder=2)
ax3a.axvline(N_total,  color=BLUE, lw=0.8, ls="--", alpha=0.25, zorder=2)
ax3a.axvspan(N_bleach, 1e5, color=RED, alpha=0.03, zorder=0)

ax3a.annotate("bleach\nevent",
    xy=(N_bleach, sigma_at_bl),
    xytext=(N_bleach*2.3, sigma_at_bl*3.0),
    fontsize=6.5, color=RED, ha="left",
    arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
ax3a.annotate(r"$N_\mathrm{total}$"+f"\n({N_total//1000}k ph)",
    xy=(N_total, sigma_at_tot),
    xytext=(N_total*1.12, sigma_at_tot*6.0),
    fontsize=6.5, color=BLUE, alpha=0.7, ha="left",
    arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6, alpha=0.55))

ax3a.axhline(floor_SMLM, color=BLUE,   lw=0.8, ls=":", alpha=0.5, zorder=1)
ax3a.axhline(floor_MF,   color=ORANGE, lw=0.8, ls=":", alpha=0.5, zorder=1)
ax3a.text(1.2, floor_SMLM*0.87,
          r"$\sigma_\mathrm{floor}^\mathrm{SMLM}$" + f" = {floor_SMLM:.0f} nm",
          fontsize=6.5, color=BLUE, alpha=0.7, va="top")
ax3a.text(1.2, floor_MF*1.12,
          r"$\sigma_\mathrm{floor}^\mathrm{MF}$" + f" = {floor_MF} nm",
          fontsize=6.5, color=ORANGE, alpha=0.7, va="bottom")

ax3a.set_xlabel("Cumulative photons collected, $N$", labelpad=3)
ax3a.set_ylabel(r"$\sigma_\mathrm{loc}$ (nm)", labelpad=3)
ax3a.set_xlim(1, 1e5); ax3a.set_ylim(0.3, 1e3)
ax3a.xaxis.set_major_formatter(
    matplotlib.ticker.LogFormatterSciNotation(base=10, labelOnlyBase=True))
ax3a.xaxis.set_minor_locator(
    matplotlib.ticker.LogLocator(base=10, subs=np.arange(2,10)*0.1, numticks=12))
ax3a.yaxis.set_minor_locator(
    matplotlib.ticker.LogLocator(base=10, subs=np.arange(2,10)*0.1, numticks=12))
ax3a.tick_params(which="minor", length=2, width=0.4)
ax3a.spines["top"].set_visible(False)
ax3a.spines["right"].set_visible(False)

handles3 = [
    Line2D([0],[0], color=BLUE,   lw=1.8, ls="-",  label=r"SMLM, $b^2=0$ ph/px"),
    Line2D([0],[0], color=BLUE,   lw=1.8, ls="--", label=r"SMLM, $b^2=10$ ph/px", alpha=0.85),
    Line2D([0],[0], color=BLUE,   lw=1.8, ls=":",  label=r"SMLM, $b^2=50$ ph/px", alpha=0.85),
    Line2D([0],[0], color=ORANGE, lw=1.8, ls="-",  label="MINFLUX"),
    Line2D([0],[0], color=RED,    lw=1.8, ls="-",  label="SMLM (early photobleach)"),
]
ax3a.legend(handles=handles3, loc="upper center", bbox_to_anchor=(0.5, -0.16),
            ncol=3, frameon=False, handlelength=1.6,
            columnspacing=1.0, labelspacing=0.35, borderaxespad=0)
ax3a.text(-0.07, 1.02, "a", transform=ax3a.transAxes,
          fontsize=11, fontweight="bold", va="top")

#  Panel b: SMLM PSF grid + MINFLUX scatter grid 
gs3b = gridspec.GridSpecFromSubplotSpec(3, 6, subplot_spec=gs3[1],
                                        hspace=0.10, wspace=0.06)

# Build 40nm bead + Gaussian PSF template (oversampled)
px_nm = 100.0; bead_d_nm = 40.0; b2_grid = 50.0
ROI = 15; oversample = 10
s_px   = s_nm / px_nm
br_px  = (bead_d_nm/2.0) / px_nm
fine   = ROI * oversample
cy_f   = cx_f = fine / 2.0
yf, xf = np.mgrid[0:fine, 0:fine] + 0.5
disk   = ((xf-cx_f)**2 + (yf-cy_f)**2 <= (br_px*oversample)**2).astype(float)
if disk.sum() == 0:
    disk[int(cy_f), int(cx_f)] = 1.0
disk  /= disk.sum()
conv   = gaussian_filter(disk, sigma=s_px*oversample)
tmpl   = conv.reshape(ROI, oversample, ROI, oversample).sum(axis=(1,3))
tmpl  /= tmpl.sum()

N_SMLM  = [50, 200, 600, 2000, 8000, 50000]
N_MF_g  = [5,  15,  50,  150,  500,  2000]
ax_half = 15.0

gs3b_pos = gs3[1].get_position(fig3)
lx_label = gs3b_pos.x0 - 0.045
fig3.text(lx_label, gs3b_pos.y0 + gs3b_pos.height*0.84,
          "Camera", va="center", ha="left", fontsize=7.5,
          style="italic", color="#333333", rotation=90)
fig3.text(lx_label, gs3b_pos.y0 + gs3b_pos.height*0.50,
          "SMLM",   va="center", ha="left", fontsize=7.5,
          style="italic", color="#333333", rotation=90)
fig3.text(lx_label, gs3b_pos.y0 + gs3b_pos.height*0.16,
          "MINFLUX",    va="center", ha="left", fontsize=7.5,
          style="italic", color="#333333", rotation=90)

for col, N in enumerate(N_SMLM):
    ax = fig3.add_subplot(gs3b[0, col])
    img = rng.poisson(N*tmpl + b2_grid).astype(float)
    ax.imshow(img, cmap="hot", interpolation="nearest", origin="lower",
              vmin=0, vmax=np.percentile(img, 99.5))
    ax.set_xticks([]); ax.set_yticks([])
    lbl = f"$N\\!={N}$" if N < 1000 else f"$N\\!=\\!{N//1000}$k"
    ax.text(0.04, 0.04, lbl, transform=ax.transAxes, fontsize=6.5,
            color="white", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.5))
    if col == 0:
        ax.text(-0.35, 1.08, "b", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

#  Middle row: SMLM position-estimate scatter (Thompson-Mortensen precision) 
# Samples drawn from N(0, sigma_TM(N, b2_grid)^2) to mirror the analytical
# treatment used in the MINFLUX row below; column N values match the top row.
for col, N in enumerate(N_SMLM):
    ax = fig3.add_subplot(gs3b[1, col])
    sig = sigma_TM(N, b2_grid)
    xy  = rng.normal(0, sig, size=(300, 2))
    ax.scatter(xy[:,0], xy[:,1], s=2.5, color=BLUE,
               alpha=0.45, linewidths=0, rasterized=True)
    ax.set_xlim(-ax_half, ax_half); ax.set_ylim(-ax_half, ax_half)
    ax.set_aspect("equal")
    ax.axhline(0, color="black", lw=0.3, alpha=0.2)
    ax.axvline(0, color="black", lw=0.3, alpha=0.2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax.spines[sp].set_color("#aaaaaa")
    sig_lbl = (f"$\\sigma\\!=\\!{sig:.1f}$nm" if sig < 10
               else f"$\\sigma\\!=\\!{sig:.0f}$nm")
    ax.text(0.97, 0.04, sig_lbl,
            transform=ax.transAxes, fontsize=6, color=BLUE,
            va="bottom", ha="right")
    lbl = f"$N\\!={N}$" if N < 1000 else f"$N\\!=\\!{N//1000}$k"
    ax.text(0.03, 0.96, lbl,
            transform=ax.transAxes, fontsize=6.5,
            color="#333333", va="top", fontweight="bold")

for col, N in enumerate(N_MF_g):
    ax = fig3.add_subplot(gs3b[2, col])
    sig = sigma_MF(N)
    xy  = rng.normal(0, sig, size=(300, 2))
    ax.scatter(xy[:,0], xy[:,1], s=2.5, color=ORANGE,
               alpha=0.45, linewidths=0, rasterized=True)
    ax.set_xlim(-ax_half, ax_half); ax.set_ylim(-ax_half, ax_half)
    ax.set_aspect("equal")
    ax.axhline(0, color="black", lw=0.3, alpha=0.2)
    ax.axvline(0, color="black", lw=0.3, alpha=0.2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax.spines[sp].set_color("#aaaaaa")
    ax.text(0.97, 0.04, f"$\\sigma\\!=\\!{sig:.1f}$nm",
            transform=ax.transAxes, fontsize=6, color=ORANGE,
            va="bottom", ha="right")
    ax.text(0.03, 0.96, f"$N\\!=\\!{N}$",
            transform=ax.transAxes, fontsize=6.5,
            color="#333333", va="top", fontweight="bold")

ax_last = fig3.axes[-1]
ax_last.plot([-ax_half*0.75, -ax_half*0.75+5], [-ax_half*0.82]*2,
             color="black", lw=1.5, solid_capstyle="butt")
ax_last.text(-ax_half*0.75+2.5, -ax_half*0.70, "5 nm",
             ha="center", va="bottom", fontsize=5.5, color="black")

fig3.savefig(OUT + "figure3.pdf", dpi=300, bbox_inches="tight")
fig3.savefig(OUT + "figure3.png", dpi=300, bbox_inches="tight")
plt.close(fig3)
print("✓ figure3 (panels a + b)")


# FIGURE 5 - Non-Gaussian linkage error distributions
# Calibrated to Wang et al. 2025 (Commun. Phys. 8, 253):
#   S (nanobody):    mean=2.5nm, SD=1.9nm, skew=0.058, kurt=2.17
#   M (primary IgG): mean=12.2nm, SD=3.6nm, skew=0.311, kurt=2.33
#   L (1+2):       mean=20.3nm, SD=4.9nm, skew=0.134, kurt=2.23, bimodal
# Plotted mean-corrected; schematic only.


COL_NANO    = "#1a9c7b"
COL_PRIMARY = "#1a5276"
COL_SECOND  = "#2e86c1"
COL_GAUSS   = "#bbbbbb"

def skewed_flat(x, sigma, skew_shift, flatten):
    x_warp = x - skew_shift * np.exp(-((x-skew_shift)**2) / (2*sigma**2))
    raw    = np.exp(-0.5 * (x_warp / sigma)**2)
    broad  = np.exp(-0.5 * (x / (sigma*1.9))**2)
    pdf    = (1 - flatten)*raw + flatten*broad
    dx     = x[1] - x[0]
    return pdf / (pdf.sum() * dx)

# Parameters tuned to match Wang et al. 2025 moments (schematic)
reporters4 = [
    # (title, colour, xlim,     sigma, skew,  flat,  bimodal, peaks)
    ("Nanobody\n(S reporter)",       COL_NANO,    (-8,  8),  1.6, -0.15, 0.25, False, None),
    ("Primary antibody\n(M reporter)", COL_PRIMARY, (-12, 12), 2.8, -1.20, 0.35, False, None),
    ("Primary + secondary\n(L reporter)", COL_SECOND, (-16, 16), 2.2,  0.00, 0.28, True, (-2.6, 3.1)),
]

fig5, axes5 = plt.subplots(1, 3, figsize=(9.5, 3.8))
fig5.patch.set_facecolor("white")

for ax, (lbl, col, xlim, sigma, skew, flat, bimodal, peaks) \
        in zip(axes5, reporters4):
    ax.set_facecolor("white")
    x  = np.linspace(xlim[0], xlim[1], 2000)
    dx = x[1] - x[0]

    if bimodal:
        c1  = skewed_flat(x - peaks[0], sigma, -0.1, 0.25)
        c2  = skewed_flat(x - peaks[1], sigma,  0.1, 0.25)
        pdf = 0.52*c1 + 0.48*c2
        pdf = pdf / (pdf.sum() * dx)
    else:
        pdf = skewed_flat(x, sigma, skew, flat)

    # Gaussian of equal variance
    std_emp = np.sqrt(np.sum(pdf * x**2 * dx))
    gauss   = np.exp(-0.5*(x/std_emp)**2)
    gauss   = gauss / (gauss.sum() * dx)

    ax.fill_between(x, pdf, color=col, alpha=0.30, zorder=2)
    ax.plot(x, pdf,   color=col,       lw=2.0, zorder=3,
            label="Schematic linkage\nerror distribution")
    ax.plot(x, gauss, color=COL_GAUSS, lw=1.4, ls="--", zorder=2,
            label="Gaussian (equal variance,\nfor comparison)")
    ax.axvline(0, color=col, lw=0.9, ls=":", alpha=0.6, zorder=1,
               label="Distribution mean")

    # Descriptor text
    if bimodal:
        descriptors = ["Broad", "Bimodal", "Platykurtic"]
    elif "M reporter" in lbl:
        descriptors = ["Intermediate width", "Left-skewed", "Platykurtic"]
    else:
        descriptors = ["Narrow", "Near-symmetric", "Platykurtic"]
    ax.text(0.97, 0.96, "\n".join(descriptors),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, style="italic", color="#555555")

    ax.set_title(lbl, fontsize=9, fontweight="bold", color=col, pad=5)
    ax.set_xlabel("Localisation offset from mean (nm)", fontsize=8)
    ax.set_ylabel("Probability density\n(schematic)", fontsize=8) \
        if ax is axes5[0] else ax.set_ylabel("")
    ax.set_xlim(xlim); ax.set_ylim(bottom=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)

# Shared legend below panels
legend_elements4 = [
    Line2D([0],[0], color="#555555",   lw=2.0,
           label="Schematic linkage error distribution"),
    Line2D([0],[0], color=COL_GAUSS,   lw=1.4, ls="--",
           label="Gaussian (equal variance, for comparison)"),
    Line2D([0],[0], color="#555555",   lw=1.0, ls=":",
           label="Distribution mean"),
]
fig5.legend(handles=legend_elements4, loc="lower center", ncol=3,
             fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=[0, 0.08, 1, 1])
fig5.savefig(OUT + "fig5_distributions.pdf", dpi=300, bbox_inches="tight")
fig5.savefig(OUT + "fig5_distributions.png", dpi=300, bbox_inches="tight")
plt.close(fig5)
print("✓ fig5_distributions")


# FIGURE 2 — Doughnut beam quality diagnostic
# Panels a–c: real beam (xy_nanoparticle_alignment_donut.tif)
# Panels d–f: synthetic ideal LG01 beam
#
# Acquisition: 640 nm, 80 mW, 8% power; 25 nm gold nanoparticle fiducial;
# 100 px/µm; deliberately calibrated asymmetric beam.
# Analysis uses luminance grayscale (consistent with RGB display in panel a).
# Display in panel a uses 4× upsampled + Gaussian-smoothed luminance for
# visual consistency with the analytically smooth theoretical beam in panel d.
# All analysis (panels b,c,e,f) runs on raw luminance data.


#  Load and prepare images
img_real  = tifffile.imread(INPUT + "xy_nanoparticle_alignment_donut.tif")
h_r, w_r  = img_real.shape[:2]
lum_real  = (0.2126*img_real[:,:,0]
             + 0.7152*img_real[:,:,1]
             + 0.0722*img_real[:,:,2])   # luminance for analysis

# Display version of real beam: upsample 4× + light smooth to match
# the analytically smooth appearance of the synthetic beam
UPSAMPLE    = 4
lum_display = zoom(lum_real, UPSAMPLE, order=1)
lum_display = gaussian_filter(lum_display, sigma=2.5)

# Synthetic LG01 beam matched to real beam peak ring radius (0.15 µm)
BQ_SCALE      = 100.0    # px per µm
CX_REAL_UM    = 0.48;  CY_REAL_UM = 0.47   # fixed physical centre (real)
R_PEAK_UM     = 0.15
h_s, w_s      = h_r, w_r
cx_s, cy_s    = w_s/2.0, h_s/2.0
CX_THEO_UM    = cx_s / BQ_SCALE
CY_THEO_UM    = cy_s / BQ_SCALE
w0_um         = R_PEAK_UM * np.sqrt(2)     # beam waist: peak at R_PEAK_UM

yi_s, xi_s = np.indices((h_s, w_s), dtype=float)
r_s        = np.sqrt((xi_s-cx_s)**2 + (yi_s-cy_s)**2) / BQ_SCALE
beam_theo  = (r_s/w0_um)**2 * np.exp(-2*r_s**2/w0_um**2)
beam_theo  = beam_theo / beam_theo.max() * 255.0

#Analysis parameters 
BQ_RINGS      = [0.10, 0.15, 0.20, 0.25]   # ring radii in µm
BQ_RING_HW    = 2.5                          # annulus half-width in px
BQ_N_BINS     = 72
bq_edges      = np.linspace(-np.pi, np.pi, BQ_N_BINS+1)
bq_centres    = (bq_edges[:-1] + bq_edges[1:]) / 2
bq_ring_cols  = [plt.cm.viridis(v) for v in [0.15, 0.40, 0.65, 0.88]]

def bq_analyse(lum, cx_um, cy_um):
    cx_px, cy_px = cx_um*BQ_SCALE, cy_um*BQ_SCALE
    h, w = lum.shape
    yi, xi = np.indices(lum.shape, dtype=float)
    r      = np.sqrt((xi-cx_px)**2 + (yi-cy_px)**2)
    theta  = np.arctan2(yi-cy_px, xi-cx_px)
    sm     = gaussian_filter(lum.astype(float), sigma=2.0)
    rp_px  = np.arange(0, min(cx_px, cy_px, w-cx_px, h-cy_px)-2, 1.0)
    rp_um  = rp_px / BQ_SCALE
    az     = np.array([sm[(r>=rb)&(r<rb+1)].mean()
                       if np.any((r>=rb)&(r<rb+1)) else 0 for rb in rp_px])
    res    = az[:4].min() / az.max() * 100
    ring_data = []
    for rp in BQ_RINGS:
        rpx = rp * BQ_SCALE
        m   = (r >= rpx-BQ_RING_HW) & (r <= rpx+BQ_RING_HW)
        if not m.any():
            ring_data.append((np.full(BQ_N_BINS, np.nan), np.nan)); continue
        rt, ri = theta[m], lum[m].astype(float)
        I_bin  = np.array([
            ri[(rt>=bq_edges[i])&(rt<bq_edges[i+1])].mean()
            if np.any((rt>=bq_edges[i])&(rt<bq_edges[i+1])) else np.nan
            for i in range(BQ_N_BINS)])
        I_norm = I_bin / np.nanmax(I_bin)
        cv     = np.nanstd(I_norm) / np.nanmean(I_norm) * 100
        ring_data.append((I_norm, cv))
    return rp_um, az, res, ring_data

rp_r, az_r, res_r, rd_r = bq_analyse(lum_real,  CX_REAL_UM,  CY_REAL_UM)
rp_t, az_t, res_t, rd_t = bq_analyse(beam_theo, CX_THEO_UM,  CY_THEO_UM)

# Figure layout 
BQ_LUT = "hot"

fig2 = plt.figure(figsize=(13, 8.4))
gs2  = gridspec.GridSpec(2, 3, figure=fig2,
                          width_ratios=[1.4, 1.4, 1.3],
                          hspace=0.50, wspace=0.42,
                          left=0.05, right=0.97,
                          top=0.95, bottom=0.07)

bq_rows = [
    ("Real beam",                        lum_display, CX_REAL_UM,  CY_REAL_UM,
     w_r, h_r, rp_r, az_r, res_r, rd_r),
    ("Theoretical beam (LG$_{01}$ mode)", beam_theo,   CX_THEO_UM,  CY_THEO_UM,
     w_s, h_s, rp_t, az_t, res_t, rd_t),
]
bq_labels = [["a","b","c"], ["d","e","f"]]
bq_axes   = {}

for ri, (title, beam_d, cx_um, cy_um, w_img, h_img,
          rp_um, az_mean, residual, ring_data) in enumerate(bq_rows):

    # Centred extent: 0,0 at beam centre
    extent   = [-cx_um, (w_img/BQ_SCALE)-cx_um,
                (h_img/BQ_SCALE)-cy_um, -cy_um]
    peak_val = az_mean.max()
    vmax     = np.percentile(beam_d, 99.5)

    #  Image panel 
    ax_a = fig2.add_subplot(gs2[ri, 0])
    bq_axes[(ri, 0)] = ax_a
    ax_a.imshow(beam_d, cmap=BQ_LUT, origin="upper", extent=extent,
                vmin=0, vmax=vmax)
    ax_a.plot(0, 0, "+", color="white", ms=12, mew=2.0, zorder=5)
    for rp, col in zip(BQ_RINGS, bq_ring_cols):
        ax_a.add_patch(Circle((0,0), rp, color=col, fill=False,
                               lw=1.3, alpha=0.9, zorder=4))
    ax_a.axhline(0, color="white", lw=0.4, alpha=0.3, ls=":")
    ax_a.axvline(0, color="white", lw=0.4, alpha=0.3, ls=":")
    ax_a.set_xlabel("x (µm)", fontsize=10)
    ax_a.set_ylabel("y (µm)", fontsize=10)
    ax_a.set_title(title, fontsize=11, pad=4)
    ax_a.tick_params(labelsize=9)
    ax_a.text(0.04, 0.97, f"zero residual = {residual:.1f}%",
              transform=ax_a.transAxes, fontsize=9, va="top",
              color="white", linespacing=1.6)
    
    #  Radial profile 
    ax_b = fig2.add_subplot(gs2[ri, 1])
    bq_axes[(ri, 1)] = ax_b
    ax_b.plot(rp_um, az_mean, color="#333333", lw=1.8)
    ax_b.axvspan(0, 0.04, color="#999999", alpha=0.15)
    ax_b.text(0.02, peak_val*0.06, "zero\nregion",
              ha="center", fontsize=8, color="#555555", linespacing=1.3)
    for rp, col in zip(BQ_RINGS, bq_ring_cols):
        ax_b.axvline(rp, color=col, lw=1.1, ls="--", alpha=0.9,
                     label=f"{rp:.2f} µm")
    ax_b.set_xlabel("Radius (µm)", fontsize=10)
    ax_b.set_ylabel("Mean intensity (a.u.)", fontsize=10)
    ax_b.tick_params(labelsize=9)
    ax_b.legend(title="Ring radius", title_fontsize=9,
                fontsize=8.5, frameon=False, loc="upper right")
    ax_b.spines[["top","right"]].set_visible(False)

    # Polar plot 
    ax_c = fig2.add_subplot(gs2[ri, 2], projection="polar")
    bq_axes[(ri, 2)] = ax_c
    ax_c.set_theta_zero_location('S')
    ax_c.spines['polar'].set_visible(False)   
    tp_ref = np.linspace(-np.pi, np.pi, 200)
    ax_c.plot(tp_ref, np.ones_like(tp_ref),
              color="red", lw=1.4, ls="--", alpha=0.75,
              label="Ideal (uniform)", zorder=1)
    for (I_norm, cv), rp, col in zip(ring_data, BQ_RINGS, bq_ring_cols):
        if np.all(np.isnan(I_norm)): continue
        tp = np.append(bq_centres, bq_centres[0])
        Ip = np.append(I_norm, I_norm[0])
        ax_c.plot(tp, Ip, color=col, lw=1.6,
                  label=f"r={rp:.2f} µm  (CV={cv:.0f}%)", zorder=3)
        ax_c.fill(tp, Ip, color=col, alpha=0.08, zorder=2)
    ax_c.set_rlim(0, 1.15)
    ax_c.set_rticks([0.25, 0.50, 0.75, 1.00])
    ax_c.set_yticklabels(["0.25","0.50","0.75","1.00"], fontsize=8)
    ax_c.set_thetagrids(np.arange(0,360,45),
                        ["0°","45°","90°","135°","180°","225°","270°","315°"],
                        fontsize=8.5)
    ax_c.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
                fontsize=8, frameon=False, ncol=2)

# Panel labels at consistent y per row (figure coords)
fig2.canvas.draw()
BQ_XOFF = -0.012;  BQ_YOFF = 0.010
for ri in range(2):
    tops    = [bq_axes[(ri,ci)].get_position().y1 for ci in range(3)]
    row_top = max(tops) + BQ_YOFF
    for ci, lbl in enumerate(bq_labels[ri]):
        pos = bq_axes[(ri,ci)].get_position()
        fig2.text(pos.x0 + BQ_XOFF, row_top, lbl,
                  fontsize=14, fontweight="bold", color="black",
                  va="bottom", ha="left", transform=fig2.transFigure)

fig2.savefig(OUT + "donut_comparison.pdf", dpi=300, bbox_inches="tight")
fig2.savefig(OUT + "donut_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
print("✓ donut_comparison (fig2 beam quality)")

print("\nAll figures saved to", OUT)
