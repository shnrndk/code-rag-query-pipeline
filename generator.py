# generator.py
import config
from openai import OpenAI

# Initialize the client using the key from config
client = OpenAI(api_key=config.OPENAI_API_KEY)

def build_prompt(query, retrieved_results):
    """
    Assembles the grounded prompt using the specific Track B format.
    """
    context_blocks = []
    
    # Chroma returns lists of lists, so we access index 0
    docs = retrieved_results['documents'][0]
    metas = retrieved_results['metadatas'][0]
    
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        citation = f"[{i+1}] repo: {meta['repo']}, path: {meta['path']}, func: {meta['func_name']}"
        context_blocks.append(f"{citation}\n{doc}")
        
    context_str = "\n\n".join(context_blocks)
    
    prompt = f"""System: Answer using only the provided code. Cite sources as repo/path::func_name.

Context:
{context_str}

User query: {query}"""
    
    return prompt

def generate_answer(prompt):
    """
    Passes the assembled prompt to OpenAI's gpt-4o-mini model.
    """
    try:
        response = client.chat.completions.create(
            model=config.GENERATOR_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Keep this low so the model stays factual and relies on the context
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"