# run_part2.py
import os
import ast
import csv
from sentence_transformers import SentenceTransformer
import chromadb
import config
import retriever
import generator

# --- QUERIES ---
TARGETED_QUERIES = [
    "How do I extract and scrape text from an HTML web page using BeautifulSoup?",
    "How do I generate an automated unit test for a python function using the OpenAI API?",
    "How to read and parse a JSON configuration file using the json module?",
    "How to write a list of dictionary data out to a CSV file automatically extracting headers?",
    "How do I execute a shell command as a subprocess with a strict timeout limit?"
]

CROSS_CORPUS_QUERIES = [
    "What is the best way to extract data from a website and save it to a file?",
    "How can I test my python code automatically?",
    "I need to read settings from a file and execute a command, how do I do that?",
    "How do I process lists of dictionaries in python?",
    "What are some ways to interact with the operating system and files?"
]

ALL_QUERIES = TARGETED_QUERIES + CROSS_CORPUS_QUERIES

def embed_new_items():
    """Reads my_custom_tools.py, extracts functions, and adds them to Chroma."""
    print("Loading custom functions into ChromaDB...")
    
    file_path = "data/my_custom_tools.py"
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    # Parse the Python file to extract individual functions
    module = ast.parse(source_code)
    new_items = []
    
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            # Get the docstring
            docstring = ast.get_docstring(node) or ""
            # Get the exact source code for the function
            func_code = ast.unparse(node)
            
            new_items.append({
                "func_name": func_name,
                "docstring": docstring,
                "code": func_code
            })

    # Prepare for Chroma
    texts = [f"{item['docstring']}\n{item['code']}" for item in new_items]
    
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embs = model.encode(texts).tolist()
    
    client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
    col = client.get_collection(config.COLLECTION_NAME)
    
    # Use IDs starting at "custom_0" to avoid colliding with the starter corpus
    col.add(
        ids=[f"custom_{i}" for i in range(len(new_items))],
        embeddings=embs,
        documents=texts,
        metadatas=[{
            'func_name': item['func_name'],
            'repo': "local/my_custom_tools", # We use this to track where it came from!
            'path': "data/my_custom_tools.py"
        } for item in new_items]
    )
    print(f"✅ Successfully added {len(new_items)} custom functions to the database!\n")

def determine_source_mix(metadatas):
    """Checks if the retrieved chunks are from the Starter corpus, New items, or Both."""
    has_starter = False
    has_new = False
    
    for meta in metadatas:
        if meta['repo'] == "local/my_custom_tools":
            has_new = True
        else:
            has_starter = True
            
    if has_starter and has_new: return "Both"
    if has_new: return "New Items"
    return "Starter Corpus"

def format_first_two_sentences(text):
    sentences = text.replace('\n', ' ').split('. ')
    if len(sentences) >= 2:
        return f"{sentences[0]}. {sentences[1]}.".strip()
    return text.strip()

def main():
    # 1. Embed the new items into the database
    embed_new_items()
    
    print("Starting Part 2 Queries...\n")
    headers = [
        "Query Type", "Query Text", "Top Sources Retrieved", "Similarity Scores", 
        "Source Mix", "Generated Answer (First 2 Sentences)", "Grounded (Yes/No)"
    ]
    
    md_lines = ["| " + " | ".join(headers) + " |", "|---|---|---|---|---|---|---|"]
    csv_data = [headers]
    
    for i, query in enumerate(ALL_QUERIES, 1):
        q_type = "Targeted" if i <= 5 else "Cross-Corpus"
        print(f"Processing {q_type} Query {i}/10: {query}")
        
        # 1. Retrieve
        results = retriever.retrieve(query)
        metas = results['metadatas'][0]
        distances = results['distances'][0] 
        
        # 2. Analyze Sources
        source_mix = determine_source_mix(metas)
        
        md_sources = "<br>".join([f"[{j+1}] {m['repo']}/{m['path']}::{m['func_name']}" for j, m in enumerate(metas)])
        csv_sources = "\n".join([f"[{j+1}] {m['repo']}/{m['path']}::{m['func_name']}" for j, m in enumerate(metas)])
        
        md_scores = "<br>".join([f"{d:.4f}" for d in distances])
        csv_scores = "\n".join([f"{d:.4f}" for d in distances])
        
        # 3. Generate
        prompt = generator.build_prompt(query, results)
        full_answer = generator.generate_answer(prompt)
        short_answer = format_first_two_sentences(full_answer)
        
        # Append to Markdown and CSV
        md_lines.append(f"| {q_type} | {query} | {md_sources} | {md_scores} | {source_mix} | {short_answer} | TBD |")
        csv_data.append([q_type, query, csv_sources, csv_scores, source_mix, short_answer, "TBD"])
        
    # Write files
    with open("results/part2_results_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    with open("results/part2_results.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(csv_data)
        
    print("\n✅ Part 2 complete!")
    print(" - Markdown table saved to 'results/part2_results_table.md'")
    print(" - Pipe-separated CSV saved to 'results/part2_results.csv'")

if __name__ == "__main__":
    main()