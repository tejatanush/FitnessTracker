from sentence_transformers import SentenceTransformer, util
def check_relevancy(text1,text2):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([text1,text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()  
    if similarity >= 0.5:
        return "Relevant"
    else:
        return "Irrelevant"