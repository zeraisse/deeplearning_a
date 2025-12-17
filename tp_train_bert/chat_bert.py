from transformers import pipeline

print("Chargement de ton IA...")
# On charge le dossier qu'on vient de créer
qa_pipeline = pipeline("question-answering", model="final_model", tokenizer="final_model", device=0)

while True:
    context = input("\nContexte : ")
    if not context: break
    question = input("Question : ")
    
    result = qa_pipeline(question=question, context=context)
    print(f"💡 Réponse : {result['answer']} (Score: {result['score']:.4f})")