import argparse
import yaml
import os
import subprocess
import sys

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.indexation import DocumentIndexer
from src.retrieval import DocumentRetriever
from src.llm import QuestionAnswerer
from src.chatbot import RAGChatbot

def load_config(config_path):
    """
    Charge la configuration depuis un fichier YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dictionnaire de configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """
    Point d'entrée principal du CLI.
    """
    parser = argparse.ArgumentParser(description="Système RAG avec LangChain")
    parser.add_argument("--config", type=str, default="config.yaml", help="Chemin vers le fichier de configuration")
    
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Commande d'indexation
    index_parser = subparsers.add_parser("index", help="Indexer des documents")
    index_parser.add_argument("--data-dir", type=str, help="Répertoire des documents")
    
    # Commande de recherche
    search_parser = subparsers.add_parser("search", help="Rechercher dans la base vectorielle")
    search_parser.add_argument("query", type=str, help="Requête de recherche")
    search_parser.add_argument("--k", type=int, default=4, help="Nombre de résultats")
    
    # Commande de question-réponse
    qa_parser = subparsers.add_parser("qa", help="Poser une question")
    qa_parser.add_argument("question", type=str, help="Question à poser")
    qa_parser.add_argument("--k", type=int, default=4, help="Nombre de documents à consulter")
    
    # Commande de chatbot
    chat_parser = subparsers.add_parser("chat", help="Démarrer une session de chat")
    
    # Commande d'interface Streamlit
    streamlit_parser = subparsers.add_parser("streamlit", help="Lancer l'interface Streamlit")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Exécuter la commande appropriée
    if args.command == "index":
        data_dir = args.data_dir or config["data_directory"]
        indexer = DocumentIndexer(
            embeddings_model_name=config["embeddings_model"],
            persist_directory=config["persist_directory"]
        )
        print(f"Indexation des documents dans {data_dir}...")
        indexer.index_documents(
            directory=data_dir,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )
        print("Indexation terminée !")
    
    elif args.command == "search":
        indexer = DocumentIndexer(
            embeddings_model_name=config["embeddings_model"],
            persist_directory=config["persist_directory"]
        )
        vectorstore = indexer.get_vectorstore()
        if vectorstore is None:
            print("Erreur: La base vectorielle n'existe pas. Veuillez d'abord indexer les documents.")
            return
        
        retriever = DocumentRetriever(vectorstore)
        print(f"Recherche pour: {args.query}")
        results = retriever.search(args.query, k=args.k)
        formatted_results = retriever.format_results(results)
        
        print("\nRésultats:")
        for i, (doc, score) in enumerate(zip(formatted_results["documents"], formatted_results["scores"])):
            print(f"\n--- Document {i+1} (Score: {score:.4f}) ---")
            print(doc["content"])
            print(f"Métadonnées: {doc['metadata']}")
    
    elif args.command == "qa":
        indexer = DocumentIndexer(
            embeddings_model_name=config["embeddings_model"],
            persist_directory=config["persist_directory"]
        )
        vectorstore = indexer.get_vectorstore()
        if vectorstore is None:
            print("Erreur: La base vectorielle n'existe pas. Veuillez d'abord indexer les documents.")
            return
        
        retriever = DocumentRetriever(vectorstore)
        qa = QuestionAnswerer(
            model_name=config["llm_model"],
            max_length=config["max_length"],
            temperature=config["temperature"]
        )
        
        print(f"Question: {args.question}")
        print("Recherche de documents pertinents...")
        results = retriever.search(args.question, k=args.k)
        context_docs = [doc for doc, _ in results]
        
        print("Génération de la réponse...")
        answer = qa.answer_question(args.question, context_docs)
        
        print("\nRéponse:")
        print(answer)
        
        print("\nDocuments utilisés:")
        for i, (doc, score) in enumerate(results):
            print(f"\n--- Document {i+1} (Score: {score:.4f}) ---")
            print(doc.page_content)
    
    elif args.command == "chat":
        indexer = DocumentIndexer(
            embeddings_model_name=config["embeddings_model"],
            persist_directory=config["persist_directory"]
        )
        vectorstore = indexer.get_vectorstore()
        if vectorstore is None:
            print("Erreur: La base vectorielle n'existe pas. Veuillez d'abord indexer les documents.")
            return
        
        qa = QuestionAnswerer(
            model_name=config["llm_model"],
            max_length=config["max_length"],
            temperature=config["temperature"]
        )
        
        chatbot = RAGChatbot(vectorstore, qa.llm)
        
        print("Chatbot RAG - Tapez 'exit' pour quitter")
        while True:
            message = input("\nVous: ")
            if message.lower() == "exit":
                break
            
            response = chatbot.chat(message)
            print(f"\nChatbot: {response}")
    
    elif args.command == "streamlit":
        # Exécuter l'application Streamlit
        streamlit_path = os.path.join(os.path.dirname(__file__), "src", "streamlit_app.py")
        subprocess.run(["streamlit", "run", streamlit_path, "--server.runOnSave", "true", "--", args.config])

    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()