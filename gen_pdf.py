from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        # Titre
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Fiche Technique : L\'Architecture Transformer', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, label):
        # Titre de section
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255) # Bleu clair
        self.cell(0, 6, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        # Corps du texte
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()

# Création du PDF
pdf = PDF()
pdf.add_page()

# Contenu
text_intro = (
    "Le Transformer (introduit par Google en 2017) est une architecture de reseau de neurones "
    "concue pour traiter des sequences (texte, audio, code) en parallele, "
    "contrairement aux anciens RNN qui lisaient mot a mot.\n\n"
    "Il est compose de deux blocs principaux : l'Encodeur (le lecteur) et le Decodeur (l'ecrivain)."
)

text_1 = (
    "Avant meme d'entrer dans les couches, le texte subit deux transformations :\n"
    "- Embedding (Plongement) : Chaque mot est transforme en un vecteur numerique dense. C'est le 'sens' brut du mot.\n"
    "- Positional Encoding : On ajoute un motif mathématique aux vecteurs pour donner l'ordre des mots (1er, 2eme...)."
)

text_2 = (
    "Role : Analyser le contexte et comprendre les relations entre les mots.\n\n"
    "A. Self-Attention (Auto-Attention)\n"
    "Chaque mot se demande : 'Quelle importance les autres mots ont-ils pour moi ?'\n"
    "Mecanisme Q, K, V (Query, Key, Value). Exemple : Dans 'La pomme est rouge', 'rouge' porte attention sur 'pomme'.\n\n"
    "B. Add & Norm\n"
    "Connexion Residuelle : On ajoute l'entree a la sortie pour que l'info circule bien.\n"
    "Normalisation : On stabilise les chiffres.\n\n"
    "C. Feed-Forward Network\n"
    "Un petit reseau de neurones classique pour digerer l'information."
)

text_3 = (
    "Role : Generer la sequence de sortie mot par mot.\n\n"
    "A. Masked Self-Attention (Attention Masquee)\n"
    "Interdiction de voir le futur. On applique un masque pour cacher les mots qui n'ont pas encore ete ecrits.\n\n"
    "B. Cross-Attention (Attention Croisee)\n"
    "Le pont entre l'Encodeur et le Decodeur. Le Decodeur regarde la phrase source complete (Encodeur) pour savoir quoi ecrire.\n\n"
    "C. Sortie (Linear + Softmax)\n"
    "Transforme le vecteur final en probabilites pour choisir le mot le plus probable dans le dictionnaire."
)

text_resume = (
    "- Encodeur seul (ex: BERT) : Pour comprendre, classer, extraire (SQuAD).\n"
    "- Decodeur seul (ex: GPT) : Pour generer du texte, chatter.\n"
    "- Encodeur-Decodeur (ex: T5) : Pour traduire, resumer.\n\n"
    "Ton projet actuel : Tu as fait du 'From Scratch' (Encodeur) pour de l'Extraction de reponse."
)

# Encodage latin-1 pour gérer les accents basiques s'ils passent, 
# sinon texte simplifié pour compatibilité maximale.
def clean(txt):
    return txt.encode('latin-1', 'replace').decode('latin-1')

pdf.chapter_title("1. L'Entree (Input)")
pdf.chapter_body(clean(text_1))

pdf.chapter_title("2. L'Encodeur (Le Lecteur)")
pdf.chapter_body(clean(text_2))

pdf.chapter_title("3. Le Decodeur (L'Ecrivain)")
pdf.chapter_body(clean(text_3))

pdf.chapter_title("Resume des differences")
pdf.chapter_body(clean(text_resume))

pdf.output("Fiche_Technique_Transformer.pdf")
print("✅ PDF généré : Fiche_Technique_Transformer.pdf")