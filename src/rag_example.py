import os
import argparse
from dotenv import load_dotenv
from rag import initialize_rag


def test_rag(query, limit):
    load_dotenv()
    rag = initialize_rag("data/agent_data.db")

    print(f"\nStructured retrieval for: '{query}'")
    print("-" * 50)
    results = rag.retrieve_relevant_interactions(query, limit)
    if not results:
        print("No results came back.")
    else:
        for i, r in enumerate(results, 1):
            print(f"Result #{i}: Session {r['session_id']}, Distance {r['similarity']:.4f}")
            print(f"Text: \"{r['transcript']}\"\n")


    print(f"\nPrompt format for: '{query}'")
    print("-" * 50)
    formatted = rag.get_relevant_memories_for_prompt(
        query, limit, prefix=f"Relevant memories for: '{query}'"
    )
    print(formatted if formatted else "No memories found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="conference speaker details")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    test_rag(args.query, args.limit)