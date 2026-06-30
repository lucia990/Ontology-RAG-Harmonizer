import argparse
from umls_search_engine import get_umls_search_engine

def main():
    parser = argparse.ArgumentParser(description="UMLS Search Engine CLI.")
    parser.add_argument("--query", type=str, required=True, help="The query string to search for.")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors to return.")
    args = parser.parse_args()

    try:
        engine = get_umls_search_engine()
        results = engine.search(args.query, args.k)
        print("\n--- Search Results ---")
        print(results.to_string(index=False)) # to_string for better console output
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all required files (model, FAISS index, metadata) are present in the correct Docker paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()