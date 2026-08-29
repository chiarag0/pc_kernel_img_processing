import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

VARIANT_ORDER = ["1ChNoConst", "1ChConst", "3ChGrid", "3ChNoGrid", "3Ch3Arrays"]

VARIANT_COLORS = {
    "1ChNoConst" : "#e74c3c",
    "1ChConst"   : "#3498db",
    "3ChGrid"    : "#2ecc71",
    "3ChNoGrid"  : "#f39c12",
    "3Ch3Arrays" : "#9b59b6"
}

VARIANT_MARKERS = {
    "1ChNoConst" : "o",
    "1ChConst"   : "s",
    "3ChGrid"    : "^",
    "3ChNoGrid"  : "D",
    "3Ch3Arrays" : "P"
}

RES_ORDER = ["360p", "720p", "1080p", "2K", "4K"]

def loadCsv(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    df = pd.read_csv(path)
    cpuDf = df[df["variant"] == "CPU"].copy()
    gpuDf = df[df["variant"] != "CPU"].copy()
    return cpuDf, gpuDf



# fixed_kernel_results.csv plots
# Fixed kernel size (11×11), scaling image resolution.
def plotFixedKernel(cpuDf, gpuDf):
    for bx, by in gpuDf[["blockX", "blockY"]].drop_duplicates().values:
        blockLabel = f"{int(bx)}x{int(by)}"
        subset = gpuDf[(gpuDf["blockX"] == bx) & (gpuDf["blockY"] == by)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Fixed Kernel (11×11) — Block size {blockLabel}",
            fontsize=13, fontweight="bold"
        )

        # Speedup vs resolution
        ax = axes[0]
        for variant in VARIANT_ORDER:
            vData = subset[subset["variant"] == variant]
            if vData.empty:
                continue
            vData = vData.set_index("resolution").reindex(RES_ORDER).dropna()
            ax.plot(
                range(len(vData)), vData["speedup"],
                label=variant,
                color=VARIANT_COLORS[variant],
                marker=VARIANT_MARKERS[variant],
                linewidth=2, markersize=7
            )
        ax.set_xticks(range(len(RES_ORDER)))
        ax.set_xticklabels(RES_ORDER)
        ax.set_xlabel("Img resolution")
        ax.set_ylabel("Speedup (CPU / GPU)")
        ax.set_title("Speedup vs Resolution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        # GPU time (ms) vs resolution
        ax = axes[1]
        for variant in VARIANT_ORDER:
            vData = subset[subset["variant"] == variant]
            if vData.empty:
                continue
            vData = vData.set_index("resolution").reindex(RES_ORDER).dropna()
            ax.plot(
                range(len(vData)), vData["gpuMs"],
                label=variant,
                color=VARIANT_COLORS[variant],
                marker=VARIANT_MARKERS[variant],
                linewidth=2, markersize=7
            )
# 
#         cpuData = cpuDf.set_index("resolution").reindex(RES_ORDER).dropna()
#         ax.plot(
#             range(len(cpuData)), cpuData["gpuMs"],
#             label="CPU", color="black",
#             linestyle="--", linewidth=1.5, marker="x", markersize=7
#         )
        ax.set_xticks(range(len(RES_ORDER)))
        ax.set_xticklabels(RES_ORDER)
        ax.set_xlabel("Img resolution")
        ax.set_ylabel("Time (ms)")
        ax.set_title("GPU time vs Resolution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        plt.tight_layout()
        outPath = os.path.join(PLOTS_DIR, f"fixed_kernel_block{blockLabel}.png")
        plt.savefig(outPath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {outPath}")

    # Overall plot: speedup for the best block size, all variants
    bestBlock = gpuDf.loc[gpuDf.groupby("variant")["speedup"].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Fixed Kernel (11×11) — Best block size for variant",
                 fontsize=13, fontweight="bold")
    for variant in VARIANT_ORDER:
        vData = gpuDf[gpuDf["variant"] == variant]
        bestB = vData.groupby(["blockX","blockY"])["speedup"].mean().idxmax()
        vData = vData[(vData["blockX"] == bestB[0]) & (vData["blockY"] == bestB[1])]
        vData = vData.set_index("resolution").reindex(RES_ORDER).dropna()
        ax.plot(
            range(len(vData)), vData["speedup"],
            label=f"{variant} ({int(bestB[0])}x{int(bestB[1])})",
            color=VARIANT_COLORS[variant],
            marker=VARIANT_MARKERS[variant],
            linewidth=2, markersize=7
        )
    ax.set_xticks(range(len(RES_ORDER)))
    ax.set_xticklabels(RES_ORDER)
    ax.set_xlabel("Img resolution")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_title("Speedup with best block size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    outPath = os.path.join(PLOTS_DIR, "fixed_kernel_best.png")
    plt.savefig(outPath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvato: {outPath}")


# fixed_resolution_results.csv plots
# Fixed image size (1080p), scaling kernel size.
def plotFixedImage(cpuDf, gpuDf):

    kernelSizes = sorted(gpuDf["kernelSize"].unique())

    for bx, by in gpuDf[["blockX", "blockY"]].drop_duplicates().values:
        blockLabel = f"{int(bx)}x{int(by)}"
        subset = gpuDf[(gpuDf["blockX"] == bx) & (gpuDf["blockY"] == by)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Fixed Resolution (1800p) — Block size {blockLabel}",
            fontsize=13, fontweight="bold"
        )

        # Speedup vs kernel size
        ax = axes[0]
        for variant in VARIANT_ORDER:
            vData = subset[subset["variant"] == variant].sort_values("kernelSize")
            if vData.empty:
                continue
            ax.plot(
                vData["kernelSize"], vData["speedup"],
                label=variant,
                color=VARIANT_COLORS[variant],
                marker=VARIANT_MARKERS[variant],
                linewidth=2, markersize=7
            )
        ax.set_xlabel("Kernel size (side)")
        ax.set_ylabel("Speedup (CPU / GPU)")
        ax.set_title("Speedup vs Kernel size")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(kernelSizes)

        # GPU time (ms) vs kernel size
        ax = axes[1]
        for variant in VARIANT_ORDER:
            vData = subset[subset["variant"] == variant].sort_values("kernelSize")
            if vData.empty:
                continue
            ax.plot(
                vData["kernelSize"], vData["gpuMs"],
                label=variant,
                color=VARIANT_COLORS[variant],
                marker=VARIANT_MARKERS[variant],
                linewidth=2, markersize=7
            )
#         cpuData = cpuDf.sort_values("kernelSize")
#         ax.plot(
#             cpuData["kernelSize"], cpuData["gpuMs"],
#             label="CPU", color="black",
#             linestyle="--", linewidth=1.5, marker="x", markersize=7
#         )
        ax.set_xlabel("Kernel size (side)")
        ax.set_ylabel("Time (ms)")
        ax.set_title("GPU time vs Kernel size")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        ax.set_xticks(kernelSizes)

        plt.tight_layout()
        outPath = os.path.join(PLOTS_DIR, f"fixed_image_block{blockLabel}.png")
        plt.savefig(outPath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Salvato: {outPath}")

    # Overall plot: best block size for each variant
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Fixed Resolution (1800p) — Best block size for variant",
                 fontsize=13, fontweight="bold")
    for variant in VARIANT_ORDER:
        vData = gpuDf[gpuDf["variant"] == variant]
        bestB = vData.groupby(["blockX","blockY"])["speedup"].mean().idxmax()
        vData = vData[
            (vData["blockX"] == bestB[0]) & (vData["blockY"] == bestB[1])
        ].sort_values("kernelSize")
        ax.plot(
            vData["kernelSize"], vData["speedup"],
            label=f"{variant} ({int(bestB[0])}x{int(bestB[1])})",
            color=VARIANT_COLORS[variant],
            marker=VARIANT_MARKERS[variant],
            linewidth=2, markersize=7
        )
    ax.set_xlabel("Kernel size (side)")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_title("Speedup with best block size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(kernelSizes)
    plt.tight_layout()
    outPath = os.path.join(PLOTS_DIR, "fixed_image_best.png")
    plt.savefig(outPath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvato: {outPath}")


    # Plot impact of block size: for each variant, compare the 3 block sizes
    fig, axes = plt.subplots(1, len(VARIANT_ORDER), figsize=(18, 4), sharey=True)
    fig.suptitle("Fixed Resolution (1800p) — Impact of block size for each variant",
                 fontsize=13, fontweight="bold")
    blockPalette = ["#e74c3c", "#3498db", "#2ecc71"]
    for idx, variant in enumerate(VARIANT_ORDER):
        ax = axes[idx]
        vData = gpuDf[gpuDf["variant"] == variant]
        for bidx, (bx, by) in enumerate(
                gpuDf[["blockX","blockY"]].drop_duplicates().values):
            bData = vData[
                (vData["blockX"] == bx) & (vData["blockY"] == by)
            ].sort_values("kernelSize")
            ax.plot(
                bData["kernelSize"], bData["speedup"],
                label=f"{int(bx)}x{int(by)}",
                color=blockPalette[bidx],
                marker="o", linewidth=2, markersize=6
            )
        ax.set_title(variant, fontsize=10)
        ax.set_xlabel("Kernel size")
        ax.set_xticks(kernelSizes)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if idx == 0:
            ax.set_ylabel("Speedup")
    plt.tight_layout()
    outPath = os.path.join(PLOTS_DIR, "fixed_image_blocksize_comparison.png")
    plt.savefig(outPath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvato: {outPath}")



if __name__ == "__main__":
    print("Plotting...\n")

    result = loadCsv("fixed_kernel_results.csv")
    if result:
        cpuDf, gpuDf = result
        print("-- fixed_kernel_results.csv --")
        plotFixedKernel(cpuDf, gpuDf)

    result = loadCsv("fixed_resolution_results.csv")
    if result:
        cpuDf, gpuDf = result
        print("\n-- fixed_resolution_results.csv --")
        plotFixedImage(cpuDf, gpuDf)

    print("\nPlots saved in results/plots/")