# run_part1.py
import retriever
import generator
import csv

# 10 natural language queries about Python functionality
QUERIES = [
    "How do I extract and scrape data from an HTML web page?",
    "How can I monitor CPU temperature, memory, or system hardware usage?",
    "How do I generate an automated unit test for a python function?",
    "How to read and parse a JSON configuration file?",
    "How do I create a concurrent thread pool for executing tasks?",
    "How to establish a connection to a SQLite database?",
    "How to write dictionary data out to a CSV file?",
    "How to send an HTTP GET request with a timeout?",
    "How do I execute a shell command as a subprocess?",
    "How do I configure basic file logging?"
]

def format_first_two_sentences(text):
    """Helper to grab just the first two sentences for the results table."""
    sentences = text.replace('\n', ' ').split('. ')
    if len(sentences) >= 2:
        return f"{sentences[0]}. {sentences[1]}.".strip()
    return text.strip()

def main():
    print("Starting Part 1 Execution...\n")
    
    # Headers for both our Markdown and CSV outputs
    headers = [
        "Query ID", 
        "Query Text", 
        "Top Sources Retrieved (repo/path::func)", 
        "Similarity Scores", 
        "Generated Answer (First 2 Sentences)", 
        "Grounded (Yes/No)"
    ]
    
    # Setup for Markdown table
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---|---|---|---|---|"
    ]
    
    # Setup for CSV data
    csv_data = [headers]
    
    for i, query in enumerate(QUERIES, 1):
        print(f"Processing Query {i}/10: {query}")
        
        # 1. Retrieve
        results = retriever.retrieve(query)
        
        # 2. Extract metadata and distances
        metas = results['metadatas'][0]
        distances = results['distances'][0] 
        
        # Markdown uses <br> for newlines in tables, CSV can use actual \n
        md_sources = "<br>".join([f"[{j+1}] {m['repo']}/{m['path']}::{m['func_name']}" for j, m in enumerate(metas)])
        csv_sources = "\n".join([f"[{j+1}] {m['repo']}/{m['path']}::{m['func_name']}" for j, m in enumerate(metas)])
        
        md_scores = "<br>".join([f"{d:.4f}" for d in distances])
        csv_scores = "\n".join([f"{d:.4f}" for d in distances])
        
        # 3. Generate
        prompt = generator.build_prompt(query, results)
        full_answer = generator.generate_answer(prompt)
        short_answer = format_first_two_sentences(full_answer)
        
        grounded = "TBD" 
        
        # Append to Markdown list
        md_row = f"| Q{i} | {query} | {md_sources} | {md_scores} | {short_answer} | {grounded} |"
        md_lines.append(md_row)
        
        # Append to CSV list
        csv_data.append([f"Q{i}", query, csv_sources, csv_scores, short_answer, grounded])
        
    # Write the Markdown output
    with open("part1_results_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    # Write the pipe-separated CSV output
    with open("part1_results.csv", "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(csv_data)
        
    print("\n✅ Part 1 complete!")
    print(" - Markdown table saved to 'part1_results_table.md'")
    print(" - Pipe-separated CSV saved to 'part1_results.csv'")

if __name__ == "__main__":
    main()