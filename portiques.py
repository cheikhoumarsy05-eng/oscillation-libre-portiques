# %% [markdown]
# # PROJET DYNA — PORTIQUES R+1 A R+4 EN OSCILLATION LIBRE
#
# Organisation du fichier :
#
# - 0) Données communes aux quatre bâtiments
# - 1) Méthode : formules et fonctions de calcul (écrites une seule fois, pour N degrés de liberté)
# - 2) Projet R+1
# - 3) Projet R+2
# - 4) Projet R+3
# - 5) Projet R+4
# - 6) Récapitulatif des quatre bâtiments

# %% [markdown]
# ## 0) Données communes
#
# Le portique R+1 est le portique de référence à deux niveaux. Les bâtiments R+2, R+3 et R+4 sont obtenus en
# ajoutant le même étage (h = 3,00 m, poteaux 0,35 × 0,35 m) au-dessus du même RDC.
#
# - RDC : hauteur 3,50 m, poteaux carrés de 0,40 m
# - Etage courant : hauteur 3,00 m, poteaux carrés de 0,35 m
# - 3 poteaux par niveau, encastrés en pied
# - Masse du sommet : 30 t ; masse de chaque autre plancher : 45 t
# - Béton de référence C25/30 : E = 31 000 MPa (Ecm de l'Eurocode 2)
# - Amortissement de référence : ξ = 5 % (Eurocode 8)
# - Oscillation libre : le sommet est déplacé de 2 cm puis relâché sans vitesse
#
# Numérotation des degrés de liberté : DDL 1 = sommet, DDL N = plancher du RDC.

# %%
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# ============ REGLAGES ============
AFFICHER_FIGURES = os.environ.get("AFFICHER_FIGURES", "1") != "0"    # False : les figures sont seulement enregistrées, sans ouvrir les fenêtres
plt.rcParams['figure.max_open_warning'] = 0

# Dossier de sauvegarde des figures : celui où se trouve ce fichier .py
try:
    dossier = os.path.dirname(os.path.abspath(__file__))
except NameError:
    dossier = os.getcwd()

# ============ DONNEES COMMUNES ============
h_rdc = 3.50          # hauteur du RDC en m
a_rdc = 0.40          # côté des poteaux du RDC en m
h_etage = 3.00        # hauteur d'un étage courant en m
a_etage = 0.35        # côté des poteaux d'un étage courant en m
n_poteaux = 3         # nombre de poteaux par niveau

m_sommet = 30000      # masse du sommet en kg (30 t)
m_plancher = 45000    # masse de chaque autre plancher en kg (45 t)

d0 = 0.02             # déplacement initial imposé au sommet en m

# Classes de béton : (fck en MPa, Ecm en N/m²) — tableau 3.1 de l'Eurocode 2
betons = {"C20/25": (20, 30e9),
          "C25/30": (25, 31e9),
          "C30/37": (30, 33e9),
          "C35/45": (35, 34e9),
          "C40/50": (40, 35e9)}
classe_ref = "C25/30"                              # béton de référence
E_ref = betons[classe_ref][1]                      # module de référence en N/m²
classes_tracees = ["C20/25", "C25/30", "C30/37"]   # classes tracées dans les figures

xi_ref = 0.05                                          # amortissement de référence (Eurocode 8)
valeurs_xi = [0.5 * xi_ref, xi_ref, 1.5 * xi_ref]      # -50 %, référence, +50 %

t = np.linspace(0, 8, 8001)    # temps de 0 à 8 secondes

# %% [markdown]
# ## 1) Méthode : formules et fonctions de calcul
#
# ### Etape 1 — Rigidité des niveaux
#
# Moment d'inertie d'un poteau carré : I = a⁴ / 12
#
# Rigidité d'un niveau (poteaux bi-encastrés, planchers infiniment rigides) : k = n · 12·E·I / h³
#
# ### Etape 2 — Matrices de masse et de rigidité
#
# Equation de l'équilibre dynamique : [M]{Ü} + [C]{U̇} + [K]{U} = {P} , avec {P} = {0} en oscillation libre
#
# Masses concentrées, donc matrice de masse diagonale : [M] = diag(m₁, m₂, …, m_N)
#
# Matrice de rigidité (tridiagonale), k₁ étant la rigidité du dernier étage et k_N celle du RDC :
#
# K₁₁ = k₁ ,  Kᵢᵢ = kᵢ₋₁ + kᵢ ,  Kᵢ,ᵢ₊₁ = Kᵢ₊₁,ᵢ = −kᵢ
#
# Pour 2 DDL on retrouve [K] = [[k₁, −k₁], [−k₁, k₁ + k₂]]
#
# ### Etape 3 — Pulsations, fréquences et périodes propres
#
# det([K] − ω²[M]) = 0  →  N pulsations propres ωᵢ ;  fᵢ = ωᵢ / 2π ;  Tᵢ = 2π / ωᵢ
#
# ### Etape 4 — Modes propres
#
# ([K] − ωᵢ²[M]) {Dᵢ} = {0} , avec le déplacement du sommet normé à 1
#
# Orthogonalité : {Dᵢ}ᵀ[M]{Dⱼ} = 0 et {Dᵢ}ᵀ[K]{Dⱼ} = 0 pour i ≠ j
#
# ### Etape 5 — Grandeurs généralisées
#
# mⱼ = {Dⱼ}ᵀ[M]{Dⱼ} ;  kⱼ = {Dⱼ}ᵀ[K]{Dⱼ} ;  contrôle : ωⱼ = √(kⱼ / mⱼ)
#
# ### Etape 6 — Amortissement
#
# Coefficient d'amortissement modal : cⱼ = 2·mⱼ·ωⱼ·ξⱼ ;  pulsation amortie : ω_Dⱼ = ωⱼ·√(1 − ξⱼ²)
#
# Amortissement de Rayleigh : [C] = α[M] + β[K] , calé sur les modes 1 et 2 :
#
# α = 2·ξ·ω₁·ω₂ / (ω₁ + ω₂) ;  β = 2·ξ / (ω₁ + ω₂) ;  ξⱼ = α / (2·ωⱼ) + β·ωⱼ / 2
#
# ### Etape 7 — Réponse en oscillation libre
#
# {U(t)} = Σ {Dᵢ}·yᵢ(t) ;  yᵢ(0) = {Dᵢ}ᵀ[M]{U₀} / mᵢ ;  ẏᵢ(0) = {Dᵢ}ᵀ[M]{U̇₀} / mᵢ
#
# Sans amortissement : yᵢ(t) = yᵢ(0)·cos(ωᵢ·t) + (ẏᵢ(0) / ωᵢ)·sin(ωᵢ·t)
#
# Avec amortissement : yᵢ(t) = e^(−ξ·ωᵢ·t) · [ yᵢ(0)·cos(ω_Dᵢ·t) + ((ẏᵢ(0) + ξ·ωᵢ·yᵢ(0)) / ω_Dᵢ)·sin(ω_Dᵢ·t) ]
#
# ### Etudes de sensibilité
#
# Béton : Ecm = 22·(fcm / 10)^0,3 en GPa, avec fcm = fck + 8 en MPa. Comme k est proportionnel à E : ω(E) / ω(E_réf) = √(E / E_réf)
#
# Amortissement : décrément logarithmique δ = 2·π·ξ / √(1 − ξ²) ; rapport de deux pics successifs uₙ₊₁ / uₙ = e^(−δ) ;
# temps pour que l'amplitude du mode 1 tombe à 5 % : t = ln(20) / (ξ·ω₁)

# %%
# ============ DEFINITION D'UN BATIMENT ============
def definir_batiment(nb_etages):
    """Construit le portique R+nb_etages. Les listes vont du sommet (indice 0) vers le RDC."""
    N = nb_etages + 1    # nombre de planchers = nombre de degrés de liberté
    masses = np.array([m_sommet] + [m_plancher] * (N - 1))    # en kg
    hauteurs = np.array([h_etage] * (N - 1) + [h_rdc])        # hauteur du niveau sous chaque plancher en m
    cotes = np.array([a_etage] * (N - 1) + [a_rdc])           # côté des poteaux de chaque niveau en m
    altitudes = np.cumsum(hauteurs[::-1])[::-1]               # altitude de chaque plancher en m
    return {"nom": "R+" + str(nb_etages), "N": N, "masses": masses,
            "hauteurs": hauteurs, "cotes": cotes, "altitudes": altitudes}


# ============ CALCUL COMPLET D'UN BATIMENT ============
def calcul_complet(bat, E_beton, xi_beton):
    """Etapes 1 à 7 pour un module E et un amortissement xi donnés."""
    N = bat["N"]

    # Etape 1 : rigidité des niveaux
    inerties = bat["cotes"]**4 / 12
    k = n_poteaux * 12 * E_beton * inerties / bat["hauteurs"]**3

    # Etape 2 : matrices de masse et de rigidité
    M = np.diag(bat["masses"]).astype(float)
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = k[i] + (k[i - 1] if i > 0 else 0)
        if i < N - 1:
            K[i, i + 1] = -k[i]
            K[i + 1, i] = -k[i]

    # Etape 3 : pulsations propres (valeurs propres de M⁻¹·K)
    valeurs, vecteurs = np.linalg.eig(np.linalg.inv(M) @ K)
    ordre = np.argsort(valeurs.real)
    omega = np.sqrt(valeurs.real[ordre])
    f = omega / (2 * math.pi)
    T = 1 / f

    # Etape 4 : modes propres, sommet normé à 1 (colonne i = mode i)
    D = vecteurs.real[:, ordre]
    D = D / D[0, :]

    # Etape 5 : grandeurs généralisées
    m_gen = np.array([D[:, j] @ M @ D[:, j] for j in range(N)])
    k_gen = np.array([D[:, j] @ K @ D[:, j] for j in range(N)])

    # Etape 6 : amortissement modal et matrice de Rayleigh (calée sur les modes 1 et 2)
    c_gen = 2 * m_gen * omega * xi_beton
    omega_D = omega * math.sqrt(1 - xi_beton**2)
    alpha = 2 * xi_beton * omega[0] * omega[1] / (omega[0] + omega[1])
    beta = 2 * xi_beton / (omega[0] + omega[1])
    C = alpha * M + beta * K
    xi_rayleigh = alpha / (2 * omega) + beta * omega / 2

    # Etape 7 : réponse en oscillation libre (sommet déplacé de d0, relâché sans vitesse)
    u0 = np.zeros(N)
    u0[0] = d0
    v0 = np.zeros(N)
    y0 = np.array([D[:, i] @ M @ u0 / m_gen[i] for i in range(N)])
    yp0 = np.array([D[:, i] @ M @ v0 / m_gen[i] for i in range(N)])
    y = np.zeros((N, len(t)))
    for i in range(N):
        y[i] = np.exp(-xi_beton * omega[i] * t) * (
            y0[i] * np.cos(omega_D[i] * t)
            + (yp0[i] + xi_beton * omega[i] * y0[i]) / omega_D[i] * np.sin(omega_D[i] * t))
    u = D @ y    # ligne 0 = sommet, dernière ligne = RDC

    return {"E": E_beton, "xi": xi_beton, "inerties": inerties, "k": k, "M": M, "K": K,
            "omega": omega, "f": f, "T": T, "D": D, "m_gen": m_gen, "k_gen": k_gen,
            "c_gen": c_gen, "omega_D": omega_D, "alpha": alpha, "beta": beta, "C": C,
            "xi_rayleigh": xi_rayleigh, "u0": u0, "y0": y0, "u": u}


# %%
# ============ SAUVEGARDE D'UNE FIGURE ============
def sauvegarder(fig, sous_dossier, nom_fichier):
    chemin = os.path.join(dossier, sous_dossier)
    os.makedirs(chemin, exist_ok=True)
    fig.savefig(os.path.join(chemin, nom_fichier), dpi=200, bbox_inches='tight')


# ============ FIGURE : SCHEMA DU PORTIQUE ============
def figure_schema(bat):
    N = bat["N"]
    z = bat["altitudes"]
    fig, ax = plt.subplots(figsize=(7, 2.2 + 1.5 * N))
    for i in range(N):
        z_bas = z[i + 1] if i < N - 1 else 0
        for x in [0, 3, 6]:    # les 3 poteaux du niveau
            ax.plot([x, x], [z_bas, z[i]], color='midnightblue', linewidth=3)
        ax.plot([-0.4, 6.4], [z[i], z[i]], color='firebrick', linewidth=6)    # plancher
        ax.plot(3, z[i], marker='o', markersize=11, color='blue')             # masse concentrée
        nom_plancher = "Sommet" if i == 0 else ("RDC" if i == N - 1 else "Etage " + str(N - 1 - i))
        ax.text(6.7, z[i], "DDL " + str(i + 1) + " (" + nom_plancher + ")\nm = " + str(round(bat["masses"][i] / 1000)) + " t",
                va='center', fontsize=11)
        ax.text(-0.7, (z_bas + z[i]) / 2, "h = " + str(bat["hauteurs"][i]) + " m\na = " + str(bat["cotes"][i]) + " m",
                va='center', ha='right', fontsize=11)
    ax.plot([-1, 7], [0, 0], color='black', linewidth=4)
    ax.text(3, -0.45, "Encastrement en pied", ha='center', va='top', fontsize=11)
    ax.set_xlim(-3.6, 10.5)
    ax.set_ylim(-1.2, z[0] + 0.8)
    ax.axis('off')
    ax.set_title("Portique " + bat["nom"] + " : " + str(N) + " degrés de liberté", fontsize=14)
    sauvegarder(fig, "Figures " + bat["nom"], "Schéma " + bat["nom"] + ".png")


# ============ FIGURE : CARTE DES MODES PROPRES ============
def figure_modes(bat, res):
    """Une case par mode, une case qui regroupe tous les modes, et (s'il reste une case)
    la déformée initiale décomposée sur les modes."""
    N = bat["N"]
    D = res["D"]
    cotes_z = np.concatenate([[0], bat["altitudes"][::-1]])    # du sol vers le sommet
    couleurs = ['green', 'orange', 'purple', 'red', 'brown']
    forme = lambda i: np.concatenate([[0], D[::-1, i]])

    nb_cases = N + 1 if (N + 1) % 2 == 0 else N + 2
    nb_colonnes = 2 if nb_cases == 4 else 3
    fig, axes = plt.subplots(2, nb_colonnes, figsize=(6.5 * nb_colonnes, 11), sharey=True)
    cases = axes.flatten()

    # Une case par mode
    for i in range(N):
        cases[i].plot(forme(i), cotes_z, marker='o', markersize=8, color=couleurs[i], linewidth=2.5)
        cases[i].set_title("Mode " + str(i + 1) + " : f = " + str(round(res["f"][i], 2)) + " Hz, T = "
                           + str(round(res["T"][i], 3)) + " s", fontsize=13)
        cases[i].set_xlabel("Déplacement normé", fontsize=12)

    # Tous les modes regroupés
    for i in range(N):
        cases[N].plot(forme(i), cotes_z, marker='o', markersize=7, color=couleurs[i], linewidth=2, label="Mode " + str(i + 1))
    cases[N].set_title("Les " + str(N) + " modes regroupés", fontsize=13)
    cases[N].set_xlabel("Déplacement normé", fontsize=12)
    cases[N].legend(fontsize=11)

    # Déformée initiale = somme des contributions des modes
    if nb_cases > N + 1:
        for i in range(N):
            cases[N + 1].plot(res["y0"][i] * forme(i) * 1000, cotes_z, marker='o', markersize=6, color=couleurs[i],
                              linewidth=1.8, linestyle='--', label="Part du mode " + str(i + 1))
        cases[N + 1].plot(np.concatenate([[0], res["u0"][::-1]]) * 1000, cotes_z, marker='o', markersize=8,
                          color='blue', linewidth=2.5, label="Déformée initiale U0")
        cases[N + 1].set_title("Déformée initiale décomposée sur les modes", fontsize=13)
        cases[N + 1].set_xlabel("Déplacement (mm)", fontsize=12)
        cases[N + 1].legend(fontsize=10)

    for ax in cases:
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_yticks(cotes_z)
        ax.grid(True)
        ax.tick_params(labelsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Hauteur (m)", fontsize=12)
    fig.suptitle("Modes propres du portique " + bat["nom"], fontsize=16)
    fig.tight_layout()
    sauvegarder(fig, "Figures " + bat["nom"], "Modes propres " + bat["nom"] + ".png")


# ============ FIGURE : CARTE A 4 CASES DES DEPLACEMENTS ============
def figure_4_cases(reponses, etiquettes, couleurs, titre, sous_dossier, nom_fichier, t_max, fond=None):
    """Une case par cas, puis une 4e case qui regroupe les trois.
    Dans chaque case, le sommet (trait plein) et le RDC (tirets) sont superposés.
    reponses : liste de 3 tableaux de déplacements (ligne 0 = sommet, dernière ligne = RDC).
    fond : réponse non amortie tracée en gris pour comparer (facultatif)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    cases = axes.flatten()
    visible = t <= t_max

    for j in range(4):
        ax = cases[j]
        if fond is not None:
            ax.plot(t[visible], fond[0][visible] * 1000, color='gray', linewidth=1, label="Sommet non amorti")
            ax.plot(t[visible], fond[-1][visible] * 1000, color='gray', linewidth=1, linestyle='--', label="RDC non amorti")
        if j < 3:
            ax.plot(t[visible], reponses[j][0][visible] * 1000, color=couleurs[j], linewidth=2.2, label="Sommet")
            ax.plot(t[visible], reponses[j][-1][visible] * 1000, color=couleurs[j], linewidth=2.2, linestyle='--', label="RDC")
            ax.set_title(etiquettes[j], fontsize=14)
        else:
            for k in range(3):
                ax.plot(t[visible], reponses[k][0][visible] * 1000, color=couleurs[k], linewidth=1.7, label=etiquettes[k] + " - sommet")
                ax.plot(t[visible], reponses[k][-1][visible] * 1000, color=couleurs[k], linewidth=1.7, linestyle='--', label=etiquettes[k] + " - RDC")
            ax.set_title("Les trois regroupés (trait plein = sommet, tirets = RDC)", fontsize=14)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize=10, ncol=2 if (j == 3 or fond is not None) else 1)
        ax.tick_params(labelsize=11)

    for ax in axes[1]:
        ax.set_xlabel("Temps (s)", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("Déplacement (mm)", fontsize=12)
    fig.suptitle(titre, fontsize=16)
    fig.tight_layout()
    sauvegarder(fig, sous_dossier, nom_fichier)


# %%
# ============ ETUDE COMPLETE D'UN BATIMENT ============
def etude_batiment(nb_etages):
    """Fait toute l'étude d'un portique : résultats de référence, figures et études de sensibilité."""
    bat = definir_batiment(nb_etages)
    nom = bat["nom"]
    N = bat["N"]
    sous_dossier = "Figures " + nom
    np.set_printoptions(suppress=True, linewidth=150)

    ref = calcul_complet(bat, E_ref, xi_ref)    # béton C25/30, xi = 5 %
    libre = calcul_complet(bat, E_ref, 0)       # même portique sans amortissement

    print("=" * 70)
    print("PORTIQUE", nom, ":", N, "degrés de liberté")
    print("=" * 70)
    print("Hauteur totale =", bat["altitudes"][0], "m ; masse totale =", bat["masses"].sum(), "kg")
    print("Rigidités des niveaux k (N/m), du dernier étage vers le RDC :", np.round(ref["k"]))
    print("Matrice de masse M (kg) :")
    print(ref["M"])
    print("Matrice de rigidité K (N/m) :")
    print(np.round(ref["K"]))
    print("Matrice d'amortissement de Rayleigh C (N.s/m), alpha =", round(ref["alpha"], 4), "1/s, beta =", round(ref["beta"], 6), "s :")
    print(np.round(ref["C"], 1))

    print()
    print(f"{'Mode':6}{'omega':>9}{'f (Hz)':>9}{'T (s)':>8}{'m gén. (kg)':>14}{'k gén. (N/m)':>15}{'c (N.s/m)':>13}{'omega_D':>10}{'y(0) (mm)':>11}")
    for i in range(N):
        print(f"{i + 1:<6}{ref['omega'][i]:9.3f}{ref['f'][i]:9.3f}{ref['T'][i]:8.3f}{ref['m_gen'][i]:14.1f}{ref['k_gen'][i]:15.0f}"
              f"{ref['c_gen'][i]:13.1f}{ref['omega_D'][i]:10.3f}{ref['y0'][i] * 1000:11.3f}")
    print("Modes propres (colonne i = mode i, ligne 1 = sommet) :")
    print(np.round(ref["D"], 4))
    print("Déplacement max du sommet =", round(np.max(np.abs(libre["u"][0])) * 1000, 2), "mm ; du RDC =",
          round(np.max(np.abs(libre["u"][-1])) * 1000, 2), "mm (sans amortissement)")

    figure_schema(bat)
    figure_modes(bat, ref)

    # ---- Sensibilité à la classe de béton (xi = 5 %) ----
    res_beton = {classe: calcul_complet(bat, E_c, xi_ref) for classe, (fck, E_c) in betons.items()}
    print()
    print(f"{'Classe':8}{'E (GPa)':>8}{'omega1':>9}{'f1 (Hz)':>9}{'T1 (s)':>8}{'écart f':>10}")
    for classe, res in res_beton.items():
        ecart = (res["omega"][0] / ref["omega"][0] - 1) * 100
        print(f"{classe:8}{res['E'] / 1e9:8.0f}{res['omega'][0]:9.3f}{res['f'][0]:9.3f}{res['T'][0]:8.3f}{ecart:8.2f} %")

    # ---- Sensibilité à l'amortissement (béton C25/30) ----
    res_xi = {xi_i: calcul_complet(bat, E_ref, xi_i) for xi_i in valeurs_xi}
    print()
    print(f"{'xi':>6}{'omega_D1':>10}{'c1 (N.s/m)':>13}{'delta':>9}{'perte/cycle':>13}{'t à 5 % (s)':>13}")
    for xi_i, res in res_xi.items():
        delta_log = 2 * math.pi * xi_i / math.sqrt(1 - xi_i**2)
        t_5 = math.log(20) / (xi_i * res["omega"][0])
        print(f"{xi_i * 100:5.1f}%{res['omega_D'][0]:10.3f}{res['c_gen'][0]:13.1f}{delta_log:9.4f}{(1 - math.exp(-delta_log)) * 100:11.1f} %{t_5:13.2f}")
    print()

    # ---- Cartes à 4 cases ----
    T1 = ref["T"][0]
    etiquettes_xi = ["xi = " + str(round(x * 100, 2)) + " %" for x in valeurs_xi]
    couleurs = ['red', 'blue', 'green']

    figure_4_cases([res_xi[x]["u"] for x in valeurs_xi], etiquettes_xi, couleurs,
                   nom + " : oscillation libre non amortie et amortie du sommet et du RDC (béton " + classe_ref + ")",
                   sous_dossier, "Oscillation libre et amortie " + nom + ".png", t_max=6 * T1, fond=libre["u"])

    figure_4_cases([res_beton[c]["u"] for c in classes_tracees], classes_tracees, couleurs,
                   nom + " : déplacements du sommet et du RDC selon la classe de béton (xi = 5 %)",
                   sous_dossier, "Sensibilité béton " + nom + ".png", t_max=6 * T1)

    figure_4_cases([res_xi[x]["u"] for x in valeurs_xi], etiquettes_xi, couleurs,
                   nom + " : déplacements du sommet et du RDC selon l'amortissement (béton " + classe_ref + ")",
                   sous_dossier, "Sensibilité amortissement " + nom + ".png", t_max=8 * T1)

    return {"bat": bat, "ref": ref, "libre": libre, "beton": res_beton, "xi": res_xi}


etudes = {}    # résultats des quatre bâtiments, utilisés dans le récapitulatif

# %% [markdown]
# ## 2) Projet R+1
#
# Portique de référence : 2 degrés de liberté (sommet et RDC).

# %%
etudes["R+1"] = etude_batiment(1)

# %% [markdown]
# Vérification à la main pour le R+1 : avec 2 DDL, det([K] − ω²[M]) = 0 est une équation du second degré en λ = ω² :
#
# m₁·m₂·λ² − [k₁·m₂ + (k₁ + k₂)·m₁]·λ + k₁·k₂ = 0
#
# et le mode i, normé à 1 au sommet, a pour composante au RDC : d₂ = (k₁ − ωᵢ²·m₁) / k₁

# %%
# ============ VERIFICATION ANALYTIQUE DU R+1 ============
k1, k2 = etudes["R+1"]["ref"]["k"]
m1, m2 = etudes["R+1"]["bat"]["masses"]

A = m1 * m2
B = -(k1 * m2 + (k1 + k2) * m1)
C0 = k1 * k2
discriminant = B**2 - 4 * A * C0
omega_main = [math.sqrt((-B - math.sqrt(discriminant)) / (2 * A)),
              math.sqrt((-B + math.sqrt(discriminant)) / (2 * A))]

for i in range(2):
    print("Mode", i + 1, ": omega à la main =", round(omega_main[i], 3), "rad/s ; omega numérique =",
          round(etudes["R+1"]["ref"]["omega"][i], 3), "rad/s ; d2 =", round((k1 - omega_main[i]**2 * m1) / k1, 4))

# %% [markdown]
# ## 3) Projet R+2
#
# Un étage courant ajouté : 3 degrés de liberté.

# %%
etudes["R+2"] = etude_batiment(2)

# %% [markdown]
# ## 4) Projet R+3
#
# Deux étages courants ajoutés : 4 degrés de liberté.

# %%
etudes["R+3"] = etude_batiment(3)

# %% [markdown]
# ## 5) Projet R+4
#
# Trois étages courants ajoutés : 5 degrés de liberté.

# %%
etudes["R+4"] = etude_batiment(4)

# %% [markdown]
# ## 6) Récapitulatif des quatre bâtiments
#
# Comparaison des quatre portiques pour le béton C25/30 et ξ = 5 %.
#
# Pour situer les résultats, on compare la période fondamentale calculée à la formule forfaitaire de l'Eurocode 8
# (EN 1998-1, 4.3.3.2.2) pour les portiques en béton armé : T = Ct·H^(3/4) , avec Ct = 0,075 et H la hauteur en m.

# %%
# ============ TABLEAU RECAPITULATIF ============
noms = list(etudes.keys())

def ligne(titre, valeurs):
    print(f"{titre:46}" + "".join(f"{v:>12}" for v in valeurs))

ligne("", noms)
ligne("Nombre de degrés de liberté", [etudes[n]["bat"]["N"] for n in noms])
ligne("Hauteur totale (m)", [etudes[n]["bat"]["altitudes"][0] for n in noms])
ligne("Masse totale (t)", [round(etudes[n]["bat"]["masses"].sum() / 1000) for n in noms])
ligne("Pulsation omega1 (rad/s)", [round(etudes[n]["ref"]["omega"][0], 3) for n in noms])
ligne("Fréquence f1 (Hz)", [round(etudes[n]["ref"]["f"][0], 3) for n in noms])
ligne("Période T1 (s)", [round(etudes[n]["ref"]["T"][0], 3) for n in noms])
ligne("Période forfaitaire EC8 (s)", [round(0.075 * etudes[n]["bat"]["altitudes"][0]**0.75, 3) for n in noms])
ligne("Fréquence f2 (Hz)", [round(etudes[n]["ref"]["f"][1], 3) for n in noms])
ligne("Masse généralisée du mode 1 (kg)", [round(etudes[n]["ref"]["m_gen"][0]) for n in noms])
ligne("Coefficient c1 (N.s/m)", [round(etudes[n]["ref"]["c_gen"][0]) for n in noms])
ligne("Part du mode 1 au sommet à t = 0 (%)", [round(etudes[n]["ref"]["y0"][0] / d0 * 100, 1) for n in noms])
ligne("Déplacement max du RDC non amorti (mm)", [round(np.max(np.abs(etudes[n]["libre"]["u"][-1])) * 1000, 2) for n in noms])
for xi_i in valeurs_xi:
    ligne("Temps pour tomber à 5 %, xi = " + str(round(xi_i * 100, 1)) + " % (s)",
          [round(math.log(20) / (xi_i * etudes[n]["ref"]["omega"][0]), 2) for n in noms])
ligne("Ecart de f1 entre C20/25 et C40/50 (%)",
      [round((etudes[n]["beton"]["C40/50"]["f"][0] / etudes[n]["beton"]["C20/25"]["f"][0] - 1) * 100, 2) for n in noms])

# %%
# ============ FIGURE : PERIODE FONDAMENTALE SELON LA HAUTEUR ============
hauteurs_totales = [etudes[n]["bat"]["altitudes"][0] for n in noms]
H_courbe = np.linspace(5, 17, 100)

fig = plt.figure(figsize=(10, 6))
plt.plot(hauteurs_totales, [etudes[n]["ref"]["T"][0] for n in noms], marker='o', markersize=9, color='blue',
         linewidth=2, label="T1 calculée (modèle brochette)")
plt.plot(H_courbe, 0.075 * H_courbe**0.75, color='red', linestyle='--', linewidth=2, label="Formule forfaitaire EC8 : 0,075·H^(3/4)")
for n, H in zip(noms, hauteurs_totales):
    plt.annotate(n, (H, etudes[n]["ref"]["T"][0]), textcoords="offset points", xytext=(-8, 10), fontsize=12)
plt.title("Période fondamentale selon la hauteur du portique", fontsize=14)
plt.xlabel("Hauteur totale (m)", fontsize=12)
plt.ylabel("Période T1 (s)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True)
sauvegarder(fig, "Figures récap", "Récap période fondamentale.png")

# %%
# ============ FIGURE : MODE FONDAMENTAL DES QUATRE BATIMENTS ============
couleurs_bat = ['green', 'blue', 'orange', 'red']

fig = plt.figure(figsize=(7, 8))
for n, couleur in zip(noms, couleurs_bat):
    cotes_z = np.concatenate([[0], etudes[n]["bat"]["altitudes"][::-1]])
    forme = np.concatenate([[0], etudes[n]["ref"]["D"][::-1, 0]])
    plt.plot(forme, cotes_z, marker='o', markersize=8, color=couleur, linewidth=2.2,
             label=n + " : T1 = " + str(round(etudes[n]["ref"]["T"][0], 3)) + " s")
plt.title("Mode fondamental des quatre portiques", fontsize=14)
plt.xlabel("Déplacement normé (sommet = 1)", fontsize=12)
plt.ylabel("Hauteur (m)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True)
sauvegarder(fig, "Figures récap", "Récap mode fondamental.png")

# %%
# ============ FIGURE : CARTE DES DEPLACEMENTS DES QUATRE BATIMENTS ============
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
visible = t <= 5
for ax, n, couleur in zip(axes.flatten(), noms, couleurs_bat):
    ax.plot(t[visible], etudes[n]["ref"]["u"][0][visible] * 1000, color=couleur, linewidth=2, label="Sommet")
    ax.plot(t[visible], etudes[n]["ref"]["u"][-1][visible] * 1000, color=couleur, linewidth=2, linestyle='--', label="RDC")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(n + " : T1 = " + str(round(etudes[n]["ref"]["T"][0], 3)) + " s", fontsize=14)
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=11)
    ax.tick_params(labelsize=11)
for ax in axes[1]:
    ax.set_xlabel("Temps (s)", fontsize=12)
for ax in axes[:, 0]:
    ax.set_ylabel("Déplacement (mm)", fontsize=12)
fig.suptitle("Oscillation libre amortie des quatre portiques (béton C25/30, xi = 5 %)", fontsize=16)
fig.tight_layout()
sauvegarder(fig, "Figures récap", "Récap déplacements.png")

# %%
# ============ AFFICHAGE DE TOUTES LES FIGURES ============
# Un seul plt.show() à la fin : toutes les fenêtres s'ouvrent en même temps.
# Les figures sont de toute façon enregistrées dans les sous-dossiers « Figures ... ».
if AFFICHER_FIGURES:
    plt.show()

# %% [markdown]
# ## Interprétation
#
# - Plus le portique est haut, plus il est souple : la période fondamentale augmente avec le nombre de niveaux.
# - Un portique à N niveaux possède N modes propres. Le mode 1 (tous les planchers dans le même sens) domine la réponse ;
#   les modes supérieurs sont plus rapides et s'amortissent plus vite, car e^(−ξ·ω·t) décroît d'autant plus vite que ω est grand.
# - La classe de béton agit seulement sur les fréquences (ω proportionnel à √E), dans la même proportion pour les quatre bâtiments.
# - L'amortissement ne change presque pas les fréquences mais commande la durée des vibrations.
