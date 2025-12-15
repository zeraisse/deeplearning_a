from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Fiche de Revision - Deep Learning & DDPM', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, f'{title}', 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10) # Reduced font size to fit more
        self.multi_cell(0, 5, body) # Reduced line height
        self.ln()

pdf = PDF()
pdf.add_page()

content = {
    "I. Le Concept DDPM": """
- Principe : Creer des images en nettoyant du bruit.
- Forward Process : On detruit l'image (Bruit Gaussien). Facile.
- Reverse Process : L'IA sculpte le bruit pour retrouver l'image. Difficile.
- U-Net : Le cerveau qui predit LE BRUIT (epsilon) a enlever, pas l'image directe.
- Loss : On compare le bruit predit vs le vrai bruit (MSE).
    """,
    "II. Les Bases Neurales": """
- ReLU : f(x) = max(0, x). Active ou non. Standard pour couches cachees.
- Sigmoid : Entre 0 et 1. Pour probabilites (Oui/Non).
- Tanh : Entre -1 et 1. Pour generation d'images (DDPM).
- Optimizer (Adam) : Le GPS qui met a jour les poids.
- Epoch : Une lecture complete du dataset.
- Batch : Paquet d'images traitees ensembles (ex: 64).
    """,
    "III. Maths & Tenseurs": """
- Tenseur : Matrice multidimensionnelle [Batch, Channels, Height, Width].
- Formule DDPM : xt = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*epsilon.
  (Melange ponderé entre image originale et bruit pur).
    """,
    "IV. Intro Encodage & Transformers (Demain)": """
- Encodage (Embedding) : Transformer les mots en vecteurs de nombres.
- Attention Mechanism : Le modele regarde toute la phrase et decide quels mots sont lies entre eux (contexte), peu importe leur distance.
    """,
    "V. Survie Technique": """
- SSH : Connexion a distance.
- Nohup/Tmux : Pour que l'entrainement survive si le PC s'eteint.
- GPU bottleneck : Si GPU a 40%, le CPU est trop lent (augmenter num_workers).
    """
}

for title, body in content.items():
    pdf.chapter_title(title)
    pdf.chapter_body(body)

pdf.output("Fiche_Revision_DeepLearning.pdf")
print("PDF généré : Fiche_Revision_DeepLearning.pdf")