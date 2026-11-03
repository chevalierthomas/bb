"""
Simulateur de bruit CD-SEM (TV scan, multi-frames).

Usage rapide :
    from simulate_cdsem import simulate_cdsem, make_line_pattern
    clean = make_line_pattern()                 # ou ta propre image propre dans [0, 1]
    noisy = simulate_cdsem(clean, n_frames=4)   # image bruitée, même échelle que clean
"""
import numpy as np
from scipy.signal import lfilter


def simulate_cdsem(clean, n_frames=4, lam_pe=2.5, se_max=2.0,
                   rel_read=0.05, rel_line=0.03, tau_bw=1.5,
                   full_scale=1.5, return_frames=False, rng=None):
    """
    Simule une acquisition CD-SEM de n_frames frames sur une image propre.

    clean       : image (H, W) dans [0, 1], interprétée comme le rendement SE relatif delta(x)/se_max
    n_frames    : nombre de frames moyennées (la dose est proportionnelle)
    lam_pe      : électrons primaires moyens par pixel et par frame (I_p * tau / e)
    se_max      : rendement d'électrons secondaires maximal (bords brillants)
    rel_read    : bruit de lecture, relatif au signal moyen d'une frame pour clean = 1
    rel_line    : offset par ligne, relatif au même signal
    tau_bw      : constante de temps du filtre RC du détecteur, en pixels (fast-scan horizontal)
    full_scale  : saturation de l'ADC, relative au signal moyen pour clean = 1
    return_frames : si True, renvoie aussi les frames individuelles (H, W, n_frames)

    Retour : image moyennée sur n_frames, à la même échelle que `clean`
             (en espérance, sortie ≈ clean flouté horizontalement).
    """
    rng = rng or np.random.default_rng()
    clean = np.asarray(clean, dtype=float)
    H, W = clean.shape

    scale = lam_pe * se_max            # signal moyen d'une frame pour clean = 1
    a = np.exp(-1.0 / tau_bw)          # filtre RC causal : y[n] = a*y[n-1] + (1-a)*x[n]

    frames = np.empty((H, W, n_frames))
    for f in range(n_frames):
        n_pe = rng.poisson(lam_pe, size=(H, W))                 # électrons primaires
        n_se = rng.poisson(clean * se_max * n_pe)               # électrons secondaires
        s = n_se + rng.normal(0, rel_read * scale, (H, W))      # bruit de lecture
        s = lfilter([1 - a], [1, -a], s, axis=1)                # bande passante, le long des lignes
        s += rng.normal(0, rel_line * scale, (H, 1))            # offset par ligne
        s = np.clip(s / scale, 0, full_scale)                   # normalisation fixe + saturation ADC
        frames[..., f] = s

    img = frames.mean(axis=2)
    return (img, frames) if return_frames else img


def make_line_pattern(H=256, W=256, pitch=32, cd=14, edge_w=2.0, base=0.25, ler=0.8, rng=None):
    """
    Image propre synthétique de lignes verticales avec bords brillants et rugosité de bord.
    Renvoie (image, bords_vrais) : bords_vrais[l] = positions x (H,) des bords gauche/droit de chaque ligne.
    """
    rng = rng or np.random.default_rng(0)
    x = np.arange(W)[None, :]
    img = np.full((H, W), base)
    edges = []
    for x0 in np.arange(pitch // 2, W - pitch // 2, pitch):
        # rugosité de bord : bruit lissé le long de y
        rough = np.convolve(rng.normal(0, ler, H + 20), np.ones(10) / 10, mode="same")[10:-10] * 3
        left = x0 - cd / 2 + rough
        right = x0 + cd / 2 + np.roll(rough, 37)
        inside = (x >= left[:, None]) & (x <= right[:, None])
        img[inside] = 0.45                                     # haut de ligne
        for e in (left, right):                                # effet de bord : pic gaussien
            img += 0.55 * np.exp(-0.5 * ((x - e[:, None]) / edge_w) ** 2)
        edges.append((left, right))
    return np.clip(img, 0, 1), edges


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    clean, edges = make_line_pattern(rng=rng)

    # 1. Série de doses
    ks = [1, 4, 16, 64]
    fig, ax = plt.subplots(1, len(ks) + 1, figsize=(16, 3.6))
    ax[0].imshow(clean[:128, :128], cmap="gray", vmin=0, vmax=1.2); ax[0].set_title("propre")
    for i, k in enumerate(ks):
        ax[i + 1].imshow(simulate_cdsem(clean, n_frames=k, rng=rng)[:128, :128], cmap="gray", vmin=0, vmax=1.2)
        ax[i + 1].set_title(f"{k} frame(s)")
    for a_ in ax: a_.axis("off")
    plt.tight_layout(); plt.savefig("dose_series.png", dpi=110)

    # 2. Paire Noise2Noise : deux réalisations indépendantes du même objet
    _, frames = simulate_cdsem(clean, n_frames=16, return_frames=True, rng=rng)
    x_in, y_tgt = frames[..., 0::2].mean(2), frames[..., 1::2].mean(2)
    a = np.exp(-1.0 / 1.5)
    expected = lfilter([1 - a], [1, -a], clean, axis=1)   # espérance = image propre floutée
    print("corrélation du bruit entrée/cible :",
          np.corrcoef((x_in - expected).ravel(), (y_tgt - expected).ravel())[0, 1].round(3),
          "(doit être ≈ 0 : condition de Noise2Noise)")

    # 3. Vérification : courbe moyenne/variance sur zone plate
    flat = np.full((256, 256), 0.5)
    for k in [1, 4, 16]:
        img = simulate_cdsem(flat, n_frames=k, rng=rng)
        print(f"k={k:2d}  moyenne={img.mean():.3f}  variance={img.var():.5f}")

    # 4. Autocorrélation du bruit, horizontale et verticale
    noise = simulate_cdsem(flat, n_frames=4, rng=rng) - 0.5
    def ac(n, axis, lag):
        return np.mean(n * np.roll(n, lag, axis=axis)) / n.var()
    print("autocorr. horizontale lag 1..3 :", [round(ac(noise, 1, l), 2) for l in (1, 2, 3)])
    print("autocorr. verticale   lag 1..3 :", [round(ac(noise, 0, l), 2) for l in (1, 2, 3)])
