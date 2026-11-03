"""Méthodes de rehaussement de contraste, une par paradigme, et outils de comparaison
pour contraste_methodes.ipynb.

Toutes les méthodes prennent une image uint8 en niveaux de gris et renvoient une
image uint8 de même taille.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from scipy.fft import dctn, idctn

import segmentation as seg


def normaliser(x, bas=0.5, haut=99.5):
    """Ramène un tableau réel sur 0..255 entre deux percentiles."""
    lo, hi = np.percentile(x, (bas, haut))
    return np.clip((x - lo) * 255 / max(hi - lo, 1e-9), 0, 255).astype(np.uint8)


# --- 1. Transformations globales (point à point) ---------------------------------

def etirement(g):
    return seg.contraster(g, "etirement")


def gamma(g, valeur=0.6):
    return seg.contraster(g, "gamma", gamma=valeur)


def egalisation(g):
    return seg.contraster(g, "egalisation")


# --- 2. Histogramme local -----------------------------------------------------------

def clahe(g, clip=2.0, tuiles=8):
    return seg.contraster(g, "clahe", clip=clip, tuiles=tuiles)


# --- 3. Éclairage × réflectance ----------------------------------------------------

def homomorphique(g, sigma=20.0, gain_bas=0.5, gain_haut=1.5):
    """log -> filtre passe-haut gaussien relevé dans Fourier -> exp.

    sigma : rayon de coupure en fréquence (en pixels du spectre).
    """
    log = np.log1p(g.astype(np.float64))
    h, w = g.shape
    v, u = np.meshgrid(np.fft.fftfreq(h) * h, np.fft.fftfreq(w) * w, indexing="ij")
    filtre = gain_bas + (gain_haut - gain_bas) * (1 - np.exp(-(u ** 2 + v ** 2) / (2 * sigma ** 2)))
    return normaliser(np.expm1(np.real(np.fft.ifft2(np.fft.fft2(log) * filtre))))


def retinex(g, sigmas=(15, 80, 250)):
    """Retinex multi-échelle : moyenne de log(I) - log(flou gaussien de I)."""
    i = g.astype(np.float64) + 1
    r = np.mean([np.log(i) - np.log(cv2.GaussianBlur(i, (0, 0), s)) for s in sigmas], axis=0)
    return normaliser(r)


# --- 4. Base + détails par filtre préservant les bords ---------------------------------

def filtre_guide(i, p, rayon, eps):
    """Filtre guidé (He et al., 2010), images réelles dans [0, 1]."""
    moy = lambda x: cv2.boxFilter(x, -1, (2 * rayon + 1,) * 2)
    mi, mp = moy(i), moy(p)
    a = (moy(i * p) - mi * mp) / (moy(i * i) - mi * mi + eps)
    b = mp - a * mi
    return moy(a) * i + moy(b)


def details_guide(g, rayon=8, eps=0.01, gain=3.0):
    """Base = filtre guidé (bords conservés), détails = image - base, amplifiés."""
    i = g.astype(np.float64) / 255
    base = filtre_guide(i, i, rayon, eps)
    return np.clip((base + gain * (i - base)) * 255, 0, 255).astype(np.uint8)


# --- 5. Multi-échelle ---------------------------------------------------------------------

def pyramide_laplacienne(g, gains=(1.2, 2.0, 2.0, 1.5)):
    """Amplifie chaque niveau de détail (du plus fin au plus grossier) puis reconstruit."""
    courant = g.astype(np.float32)
    details = []
    for _ in gains:
        bas = cv2.pyrDown(courant)
        details.append(courant - cv2.pyrUp(bas, dstsize=courant.shape[::-1]))
        courant = bas
    for d, k in zip(reversed(details), reversed(gains)):
        courant = cv2.pyrUp(courant, dstsize=d.shape[::-1]) + k * d
    return np.clip(courant, 0, 255).astype(np.uint8)


def ondelettes(g, gains=(1.2, 2.0, 2.0), ondelette="db2"):
    """Amplifie les coefficients de détail (du plus fin au plus grossier) puis reconstruit."""
    coeffs = pywt.wavedec2(g.astype(np.float64), ondelette, level=len(gains))
    # coeffs = [approximation, détails grossiers, ..., détails fins]
    for j, k in enumerate(reversed(gains), start=1):
        coeffs[j] = tuple(k * c for c in coeffs[j])
    return np.clip(pywt.waverec2(coeffs, ondelette)[:g.shape[0], :g.shape[1]], 0, 255).astype(np.uint8)


# --- 6. Morphologie mathématique ---------------------------------------------------------

def tophat(g, rayon=6):
    """I + top-hat blanc - top-hat noir : rehausse les structures plus petites que le rayon."""
    blanc = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, seg.disque(rayon)).astype(np.int16)
    noir = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, seg.disque(rayon)).astype(np.int16)
    return np.clip(g + blanc - noir, 0, 255).astype(np.uint8)


# --- 7. Optimisation : domaine du gradient (Poisson) -----------------------------------

def gradient_poisson(g, beta=0.8, sigma=2.0):
    """Remodèle le champ de gradients puis cherche l'image la plus proche (Fattal et al., 2002).

    Chaque gradient est multiplié par (|∇| / moyenne)^(beta - 1) : beta < 1 renforce les
    gradients faibles par rapport aux forts. L'image u minimise ||∇u - G||², résolu
    exactement (équation de Poisson, bords de Neumann) par transformée en cosinus.
    """
    i = g.astype(np.float64)
    gx = np.diff(i, axis=1, append=i[:, -1:])
    gy = np.diff(i, axis=0, append=i[-1:, :])
    norme = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), sigma) + 1e-6
    facteur = (norme / norme.mean()) ** (beta - 1)
    gx, gy = gx * facteur, gy * facteur
    div = np.diff(gx, axis=1, prepend=0) + np.diff(gy, axis=0, prepend=0)
    h, w = g.shape
    ky, kx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    denom = (2 * np.cos(np.pi * kx / w) - 2) + (2 * np.cos(np.pi * ky / h) - 2)
    denom[0, 0] = 1
    u_hat = dctn(div, norm="ortho") / denom
    u_hat[0, 0] = 0
    return normaliser(idctn(u_hat, norm="ortho"))


# --- 8. Fusion d'expositions -----------------------------------------------------------

def fusion_expositions(g, gammas=(0.5, 1.0, 2.0)):
    """Fusion de Mertens : par pixel, poids aux versions bien exposées et contrastées."""
    poses = [cv2.cvtColor((255 * (g / 255.0) ** k).astype(np.uint8), cv2.COLOR_GRAY2BGR) for k in gammas]
    fusion = cv2.createMergeMertens().process(poses)
    return normaliser(cv2.cvtColor(fusion, cv2.COLOR_BGR2GRAY))


# --- 9. Logique floue -----------------------------------------------------------------

def floue(g, iterations=1):
    """Intensification de Pal et King : appartenance μ poussée vers 0 ou 1 autour de 0,5."""
    mu = (g - g.min()) / max(float(g.max()) - float(g.min()), 1.0)
    for _ in range(iterations):
        mu = np.where(mu < 0.5, 2 * mu ** 2, 1 - 2 * (1 - mu) ** 2)
    return (mu * 255).astype(np.uint8)


# --- 10. Apprentissage profond non supervisé : Zero-DCE ---------------------------------

class DCENet(nn.Module):
    """DCE-Net (Guo et al., 2020) pour une image en gris : n_courbes cartes α par pixel."""

    def __init__(self, n_courbes=8, canaux=32):
        super().__init__()
        c = canaux
        self.e = nn.ModuleList([nn.Conv2d(1, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1),
                                nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1),
                                nn.Conv2d(2 * c, c, 3, padding=1), nn.Conv2d(2 * c, c, 3, padding=1),
                                nn.Conv2d(2 * c, n_courbes, 3, padding=1)])

    def forward(self, x):
        x1 = F.relu(self.e[0](x))
        x2 = F.relu(self.e[1](x1))
        x3 = F.relu(self.e[2](x2))
        x4 = F.relu(self.e[3](x3))
        x5 = F.relu(self.e[4](torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e[5](torch.cat([x2, x5], 1)))
        alphas = torch.tanh(self.e[6](torch.cat([x1, x6], 1)))
        for k in range(alphas.shape[1]):          # courbe quadratique itérée LE(I) = I + α I (1 - I)
            a = alphas[:, k:k + 1]
            x = x + a * x * (1 - x)
        return x, alphas


def zero_dce(g, iterations=100, exposition=0.5, graine=0):
    """Zero-DCE entraîné sur l'image elle-même (à demi-résolution), sans référence.

    Pertes : cohérence spatiale (garder les différences entre régions voisines),
    exposition (luminance locale vers `exposition`), régularité des cartes α.
    """
    torch.manual_seed(graine)
    plein = torch.from_numpy(g.astype(np.float32) / 255)[None, None]
    x = F.interpolate(plein, scale_factor=0.5, mode="area")      # réseau convolutif : s'applique ensuite en plein
    reseau = DCENet()
    optimiseur = torch.optim.Adam(reseau.parameters(), lr=1e-3)
    noyaux = torch.tensor([[[0, 0, 0], [-1, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
                           [[0, -1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, -1, 0]]],
                          dtype=torch.float32)[:, None]
    for _ in range(iterations):
        y, alphas = reseau(x)
        px, py = F.avg_pool2d(x, 4), F.avg_pool2d(y, 4)
        spatiale = ((F.conv2d(py, noyaux, padding=1) - F.conv2d(px, noyaux, padding=1)) ** 2).sum(1).mean()
        expo = (F.avg_pool2d(y, 8) - exposition).abs().mean()
        tv = (alphas.diff(dim=2) ** 2).mean() + (alphas.diff(dim=3) ** 2).mean()
        perte = spatiale + 10 * expo + 200 * tv
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
    with torch.no_grad():
        return (reseau(plein)[0][0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)


# --- Mesures et affichage ------------------------------------------------------------

def mesures(g):
    """Valeurs clés d'une image.

    contraste_rms : écart-type des niveaux de gris ;
    dynamique : écart entre percentiles 1 et 99 ;
    entropie : information de l'histogramme (bits, max 8) ;
    nettete : norme moyenne du gradient de Sobel ;
    bruit : écart-type estimé (Immerkær) ;
    nettete/bruit : > 1 et en hausse = les bords gagnent plus que le bruit.
    """
    p = np.bincount(g.ravel(), minlength=256) / g.size
    p = p[p > 0]
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1)
    nettete = float(np.hypot(gx, gy).mean())
    bruit = seg.estimer_bruit(g)
    lo, hi = np.percentile(g, (1, 99))
    return {"contraste_rms": float(g.std()), "dynamique": float(hi - lo), "entropie": float(-(p * np.log2(p)).sum()),
            "nettete": nettete, "bruit": bruit, "nettete/bruit": nettete / max(bruit, 1e-9)}


GRIS, BLEU = "#8a8a8a", "#2b6cb0"


def comparer(g, sortie, titre):
    """Affiche original | résultat | histogrammes, et les valeurs clés. Renvoie les mesures."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    for a, img, t in ((ax[0], g, "Original"), (ax[1], sortie, titre)):
        a.imshow(img, cmap="gray", vmin=0, vmax=255)
        a.set_title(t, fontsize=10)
        a.axis("off")
    bins = np.arange(257)
    ax[2].hist(g.ravel(), bins, histtype="stepfilled", color=GRIS, alpha=0.35, label="Original")
    ax[2].hist(sortie.ravel(), bins, histtype="step", color=BLEU, linewidth=1.5, label=titre)
    ax[2].set_xlim(0, 255)
    ax[2].set_yticks([])
    ax[2].set_xlabel("niveau de gris", fontsize=9)
    ax[2].legend(fontsize=8, frameon=False)
    for cote in ("top", "right", "left"):
        ax[2].spines[cote].set_visible(False)
    plt.tight_layout()
    plt.show()
    m = mesures(sortie)
    display(pd.DataFrame({"Original": mesures(g), titre: m}).T.round(2))
    return m


# --- Pistes pour préparer l'entrée de SAM ----------------------------------------------

def nlm(g):
    return seg.debruiter(g, "nlm")


def nlm_clahe(g, clip=2.0, tuiles=8):
    """Débruiter d'abord (NLM), puis CLAHE : le CLAHE n'amplifie plus le grain."""
    return clahe(nlm(g), clip, tuiles)


def perona_malik(g, iterations=15, kappa=15.0, pas=0.2):
    """Diffusion anisotrope : lisse là où le gradient est < kappa, s'arrête aux bords."""
    u = g.astype(np.float64)
    for _ in range(iterations):
        d = [np.roll(u, -1, 0) - u, np.roll(u, 1, 0) - u, np.roll(u, -1, 1) - u, np.roll(u, 1, 1) - u]
        u = u + pas * sum(di * np.exp(-(di / kappa) ** 2) for di in d)
    return np.clip(u, 0, 255).astype(np.uint8)


def specification(g, reference):
    """Spécification d'histogramme : donne à g l'histogramme de reference.

    Utile pour que toutes les images d'un jeu (entraînement et test) aient la même
    distribution de gris, quelle que soit l'exposition au MEB.
    """
    cdf_g = np.cumsum(np.bincount(g.ravel(), minlength=256)) / g.size
    cdf_r = np.cumsum(np.bincount(reference.ravel(), minlength=256)) / reference.size
    return np.searchsorted(cdf_r, cdf_g).clip(0, 255).astype(np.uint8)[g]


def composite_rgb(g):
    """Trois vues dans les trois canaux d'entrée de SAM : brute, NLM, NLM + CLAHE."""
    d = nlm(g)
    return np.dstack([g, d, clahe(d)])


def grille_clahe(g, clips=(1.0, 2.0, 4.0), tuiles=(4, 8, 16), debruiter=False):
    """Mosaïque du CLAHE pour plusieurs clip × tuiles, et rapport netteté/bruit de chacun."""
    base = nlm(g) if debruiter else g
    fig, ax = plt.subplots(len(clips), len(tuiles), figsize=(3 * len(tuiles), 3 * len(clips)))
    table = {}
    for i, c in enumerate(clips):
        for j, t in enumerate(tuiles):
            sortie = clahe(base, c, t)
            ratio = mesures(sortie)["nettete/bruit"]
            table[(c, t)] = ratio
            ax[i, j].imshow(sortie, cmap="gray", vmin=0, vmax=255)
            ax[i, j].set_title(f"clip {c:g}, tuiles {t} — n/b {ratio:.1f}", fontsize=9)
            ax[i, j].axis("off")
    plt.tight_layout()
    plt.show()
    return pd.Series(table, name="nettete/bruit").unstack().rename_axis(index="clip", columns="tuiles").round(2)


# --- Effet sur SAM ---------------------------------------------------------------------

def effet_sam(g, variantes, labels=None, pas=48, decalage=6, identifiant="facebook/sam2.1-hiera-small"):
    """Compare des prétraitements par ce qu'en fait SAM (invites = grille de points).

    score_sam : IoU prédit par SAM lui-même (sa confiance) ;
    controles_ok : part des masques passant les contrôles SegSEM ;
    stabilite : IoU entre les masques de deux points voisins (± decalage px) du même
                objet, sans vérité terrain : un bon prétraitement donne le même objet ;
    iou_verite : IoU avec les annotations (labels), si fournies — seule mesure décisive.
    """
    import segmentation_sam as sam
    import segsem

    modele = sam.charger(identifiant)
    rng = np.random.default_rng(0)
    h, w = g.shape
    points = [(x, y) for y in range(pas // 2, h, pas) for x in range(pas // 2, w, pas)]
    lignes = {}
    for nom, f in variantes.items():
        image = f(g)
        enc = sam.encoder(*modele, image)

        def meilleur(p):
            m, s = sam.predire(*modele, enc, [p], [1])
            return m[s.argmax()], float(s.max())

        scores, ok, stab = [], [], []
        for x, y in points:
            m, s = meilleur((x, y))
            scores.append(s)
            ok.append(not segsem.controler(m, s))
            voisin = (np.clip(x + decalage, 0, w - 1), np.clip(y + decalage, 0, h - 1))
            if m[voisin[1], voisin[0]]:                       # même objet : le masque doit être le même
                m2, _ = meilleur(voisin)
                stab.append((m & m2).sum() / max((m | m2).sum(), 1))
        ligne = {"score_sam": np.mean(scores), "controles_ok": np.mean(ok), "stabilite": np.mean(stab)}
        if labels is not None:
            ious = []
            for r in range(1, labels.max() + 1):
                v = labels == r
                if v.sum() >= 30:
                    for _ in range(3):
                        m, _ = meilleur(segsem.tirer_point(v, rng))
                        ious.append((m & v).sum() / (m | v).sum())
            ligne["iou_verite"] = np.mean(ious)
        lignes[nom] = ligne
    return pd.DataFrame(lignes).T.round(3)


def segmenter_auto(modele, image, pas=32, aire_max=0.5, score_min=0.7):
    """SAM seul sur une grille de points. Pour chaque point non couvert, garde le
    candidat de meilleur score parmi ceux de moins de aire_max × l'image (sinon SAM
    prend souvent tout le fond d'un bloc) et d'au moins score_min (en dessous, ce
    sont des fragments de fond). Renvoie (labels, scores des objets retenus).
    """
    import segmentation_sam as sam

    h, w = image.shape[:2]
    enc = sam.encoder(*modele, image)
    labels = np.zeros((h, w), np.int32)
    scores = []
    for y in range(pas // 2, h, pas):
        for x in range(pas // 2, w, pas):
            if labels[y, x]:
                continue
            masques, s = sam.predire(*modele, enc, [(x, y)], [1])
            valides = [i for i in np.argsort(s)[::-1] if s[i] >= score_min and 30 <= masques[i].sum() <= aire_max * h * w]
            if not valides:
                continue
            nouveau = masques[valides[0]] & (labels == 0)
            if nouveau.sum() >= 30:
                labels[nouveau] = labels.max() + 1
                scores.append(float(s[valides[0]]))
    return labels, scores


def mosaique(g, panneaux, colonnes=3):
    """Affiche {titre: labels} superposés à g."""
    n = len(panneaux)
    colonnes = min(n, colonnes)
    fig, ax = plt.subplots(-(-n // colonnes), colonnes, figsize=(5 * colonnes, 5 * -(-n // colonnes)), squeeze=False)
    for a, (titre, labels) in zip(ax.ravel(), panneaux.items()):
        a.imshow(seg.superposer(g, labels, alpha=0.45))
        a.set_title(titre, fontsize=10)
        a.axis("off")
    for a in ax.ravel()[n:]:
        a.axis("off")
    plt.tight_layout()
    plt.show()


def voir_sam(g, variantes, pas=32, aire_max=0.5, score_min=0.7, identifiant="facebook/sam2.1-hiera-small"):
    """Segmentation automatique par SAM seul (voir segmenter_auto) pour chaque
    prétraitement, superposée à l'image d'origine."""
    import segmentation_sam as sam

    modele = sam.charger(identifiant)
    panneaux = {}
    for nom, f in variantes.items():
        labels, scores = segmenter_auto(modele, f(g), pas, aire_max, score_min)
        panneaux[f"{nom} — {labels.max()} objets, score moyen {np.mean(scores or [0]):.2f}"] = labels
    mosaique(g, panneaux)


def comparer_modeles(g, pretraitement, labels=None, modeles=None, pas=32, aire_max=0.5, score_min=0.7):
    """Même prétraitement, plusieurs tailles de SAM : masques côte à côte et tableau.

    temps_encodage : coût de l'étape lourde, une fois par image ;
    iou_verite : IoU avec les annotations (3 points tirés par objet), si labels fourni.
    """
    import gc
    import time
    import segmentation_sam as sam
    import segsem

    modeles = modeles or sam.MODELES
    image = pretraitement(g)
    panneaux, lignes = {}, {}
    for k, (nom, identifiant) in enumerate(modeles.items()):
        modele = sam.charger(identifiant)
        if k == 0:                      # échauffement : le 1er passage de PyTorch est plus lent
            sam.encoder(*modele, image)
        t = time.time()
        enc = sam.encoder(*modele, image)
        duree = time.time() - t
        auto, scores = segmenter_auto(modele, image, pas, aire_max, score_min)
        ligne = {"temps_encodage_s": duree, "objets": auto.max(), "score_moyen": np.mean(scores or [0])}
        if labels is not None:
            rng, ious = np.random.default_rng(0), []
            for r in range(1, labels.max() + 1):
                v = labels == r
                if v.sum() >= 30:
                    for _ in range(3):
                        m, s = sam.predire(*modele, enc, [segsem.tirer_point(v, rng)], [1])
                        m = m[s.argmax()]
                        ious.append((m & v).sum() / (m | v).sum())
            ligne["iou_verite"] = np.mean(ious)
        lignes[nom] = ligne
        panneaux[f"{nom} — {auto.max()} objets, score moyen {ligne['score_moyen']:.2f}"] = auto
        del modele, enc
        gc.collect()
    mosaique(g, panneaux)
    return pd.DataFrame(lignes).T.round(3)



# --- Invites données par une méthode classique ------------------------------------------

def regions_watershed(g):
    """Régions du watershed sur bords brillants, réglages par défaut de l'app."""
    d = seg.debruiter(g, "gaussien", 2.0)
    tophat = seg.carte_bords(d, 6)
    bords, _ = seg.masque_bords(tophat)
    return seg.renumeroter(seg.watershed_bords(d, bords, tophat, 8, 200))


def invites_regions(regions):
    """Pour chaque région : point le plus intérieur (max de la distance au bord) et boîte englobante.

    Triées de la plus petite à la plus grande : les objets sont pris avant le fond,
    dont la boîte couvre souvent toute l'image (SAM y engloberait les objets).
    """
    invites = []
    aires = np.bincount(regions.ravel(), minlength=regions.max() + 1)
    for r in sorted(range(1, regions.max() + 1), key=lambda r: aires[r]):
        m = (regions == r).astype(np.uint8)
        if m.sum() < 30:
            continue
        y, x = np.unravel_index(cv2.distanceTransform(m, cv2.DIST_L2, 5).argmax(), m.shape)
        ys, xs = np.nonzero(m)
        invites.append(((float(x), float(y)), (xs.min(), ys.min(), xs.max(), ys.max())))
    return invites


def sam_guide(modele, image, invites, boite=True):
    """SAM avec une invite par région classique (point, ou point + boîte)."""
    import segmentation_sam as sam

    enc = sam.encoder(*modele, image)
    labels = np.zeros(image.shape[:2], np.int32)
    scores = []
    for point, b in invites:
        masques, s = sam.predire(*modele, enc, [point], [1], b if boite else None)
        nouveau = masques[s.argmax()] & (labels == 0)
        if nouveau.sum() >= 30:
            labels[nouveau] = labels.max() + 1
            scores.append(float(s.max()))
    return labels, scores


def iou_objets(prediction, verite):
    """Pour chaque objet annoté, IoU avec la région prédite qui le recouvre le mieux ; moyenne."""
    ious = []
    for r in range(1, verite.max() + 1):
        v = verite == r
        if v.sum() < 30:
            continue
        ids, inter = np.unique(prediction[v], return_counts=True)
        meilleur = 0.0
        for i, n in zip(ids, inter):
            if i:
                meilleur = max(meilleur, n / (v.sum() + (prediction == i).sum() - n))
        ious.append(meilleur)
    return float(np.mean(ious)) if ious else np.nan


def comparer_invites(g, pretraitement, labels=None, identifiant="facebook/sam2.1-hiera-small", pas=32):
    """Watershed seul, SAM sur grille, SAM guidé par les régions du watershed (points, puis boîtes + points)."""
    import segmentation_sam as sam

    modele = sam.charger(identifiant)
    image = pretraitement(g)
    regions = regions_watershed(g)
    invites = invites_regions(regions)
    resultats = {"Watershed seul": (regions, [])}
    resultats["SAM, grille de points"] = segmenter_auto(modele, image, pas)
    resultats["SAM guidé : points"] = sam_guide(modele, image, invites, boite=False)
    resultats["SAM guidé : boîtes + points"] = sam_guide(modele, image, invites, boite=True)
    lignes, panneaux = {}, {}
    for nom, (lab, scores) in resultats.items():
        ligne = {"objets": int(lab.max()), "score_moyen": np.mean(scores) if scores else np.nan}
        if labels is not None:
            ligne["iou_verite"] = iou_objets(lab, labels)
        lignes[nom] = ligne
        panneaux[f"{nom} — {lab.max()} objets"] = lab
    mosaique(g, panneaux, colonnes=2)
    return pd.DataFrame(lignes).T.round(3)


# --- Contours précis -------------------------------------------------------------------

def lignes_niveau(z, niveau, principale=True):
    """Lignes de niveau d'une carte réelle, interpolées entre pixels : liste de (N, 2) en (x, y).

    principale : écarter les îlots parasites (lignes de moins de 20 % de la plus longue).
    Un objet qui touche le bord de l'image a un contour ouvert en plusieurs morceaux,
    d'où le seuil relatif plutôt que « la plus longue seulement ».
    """
    import contourpy
    lignes = [l for l in contourpy.contour_generator(z=z).lines(niveau) if len(l) > 2]
    if principale and lignes:
        longueurs = [np.linalg.norm(np.diff(l, axis=0), axis=1).sum() for l in lignes]
        lignes = [l for l, n in zip(lignes, longueurs) if n >= 0.2 * max(longueurs)]
    return lignes


def largeur_a(lignes, y):
    """Largeur (px) à l'ordonnée y : écart entre la première et la dernière traversée du contour."""
    xs = []
    for l in lignes:
        p, q = l[:-1], l[1:]
        croise = ((p[:, 1] - y) * (q[:, 1] - y) <= 0) & (p[:, 1] != q[:, 1])
        t = (y - p[croise, 1]) / (q[croise, 1] - p[croise, 1])
        xs.extend(p[croise, 0] + t * (q[croise, 0] - p[croise, 0]))
    return max(xs) - min(xs) if len(xs) >= 2 else np.nan


def ecart_moyen(lignes_a, lignes_b):
    """Distance moyenne (px) des points de a au contour b."""
    from scipy.spatial import cKDTree
    if not lignes_a or not lignes_b:
        return np.nan
    return float(cKDTree(np.vstack(lignes_b)).query(np.vstack(lignes_a))[0].mean())


def contours_precis(g, pretraitement, objet=None, labels=None, marge=16,
                    identifiant="facebook/sam2.1-hiera-small"):
    """Trois contours du même objet : masque binaire, logits sous-pixel, logits sous-pixel après zoom.

    SAM prédit en 256 × 256 : sur une image de 480 px, un pixel de masque couvre ~2 px.
    - sous-pixel : ligne de niveau 0 des logits (interpolation) au lieu du masque seuillé ;
    - zoom : SAM n'encode que la boîte de l'objet (+ marge), agrandie à 1024 px.
    objet : numéro de région du watershed (défaut : la plus grande de moins de la moitié de l'image).
    """
    import segmentation_sam as sam

    modele = sam.charger(identifiant)
    image = pretraitement(g)
    regions = regions_watershed(g)
    if objet is None:
        aires = np.bincount(regions.ravel())
        aires[0] = 0
        aires[aires > g.size / 2] = 0
        objet = int(aires.argmax())
    (point, boite), = invites_regions((regions == objet).astype(np.int32))
    x0, y0, x1, y1 = boite

    logits, s = sam.predire(*modele, sam.encoder(*modele, image), [point], [1], boite, binariser=False)
    plein = logits[s.argmax()]

    h, w = g.shape
    cx0, cy0 = max(x0 - marge, 0), max(y0 - marge, 0)
    cx1, cy1 = min(x1 + marge + 1, w), min(y1 + marge + 1, h)
    rogne = image[cy0:cy1, cx0:cx1]
    logits_z, s_z = sam.predire(*modele, sam.encoder(*modele, rogne), [(point[0] - cx0, point[1] - cy0)], [1],
                                (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0), binariser=False)
    decalage = np.array([cx0, cy0])

    contours = {
        "Masque binaire (plein)": lignes_niveau((plein > 0).astype(float), 0.5),
        "Sous-pixel (plein)": lignes_niveau(plein, 0.0),
        "Sous-pixel (zoom)": [l + decalage for l in lignes_niveau(logits_z[s_z.argmax()], 0.0)],
    }
    verite = None
    if labels is not None:
        ids, n = np.unique(labels[regions == objet], return_counts=True)
        n[ids == 0] = 0
        if n.max():
            verite = lignes_niveau((labels == ids[n.argmax()]).astype(float), 0.5)

    couleurs = ["#e8a33d", "#2b6cb0", "#c0392b"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    yc = (y0 + y1) / 2
    # détail : carré de 30 px centré sur le bord gauche de l'objet, sur la ligne de mesure
    pz = np.vstack(contours["Sous-pixel (zoom)"])
    proches = pz[np.abs(pz[:, 1] - yc) < 2]
    centre = proches[proches[:, 0].argmin()] if len(proches) else pz[0]
    for a, (ax0, ay0, ax1, ay1), titre in ((ax[0], (cx0, cy0, cx1, cy1), f"Objet {objet}"),
                                           (ax[1], (*(centre - 15), *(centre + 15)), "Détail (bord gauche, ligne de mesure)")):
        a.imshow(g, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        for (nom, lignes), c in zip(contours.items(), couleurs):
            for k, l in enumerate(lignes):
                a.plot(l[:, 0], l[:, 1], color=c, linewidth=1.5, label=nom if k == 0 else None)
        if verite:
            for k, l in enumerate(verite):
                a.plot(l[:, 0], l[:, 1], color="#2e9e5b", linewidth=1.5, linestyle="--",
                       label="Annotation" if k == 0 else None)
        a.set_xlim(ax0, ax1)
        a.set_ylim(ay1, ay0)
        a.set_title(titre, fontsize=10)
        a.axis("off")
    ax[0].axhline(yc, color="white", linewidth=0.8, linestyle=":")
    ax[1].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.show()

    lignes = {}
    for nom, l in contours.items():
        ligne = {f"largeur_y{yc:.0f}_px": largeur_a(l, yc),
                 "ecart_au_zoom_px": ecart_moyen(l, contours["Sous-pixel (zoom)"])}
        if verite:
            ligne["ecart_annotation_px"] = (ecart_moyen(l, verite) + ecart_moyen(verite, l)) / 2
        lignes[nom] = ligne
    return pd.DataFrame(lignes).T.round(2)
