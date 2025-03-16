# streamlit_app.py (Mise à jour pour Llama3.1 uniquement)

import streamlit as st
import os
import yaml
import subprocess
import time
import requests
import warnings
from typing import Dict, Any, List, Optional

# Supprimer les avertissements de dépréciation pour les classes que nous utilisons
warnings.filterwarnings("ignore", message="The class `Ollama` was deprecated")
warnings.filterwarnings("ignore", message="This class is deprecated.")
warnings.filterwarnings("ignore", message="The method `Chain.run` was deprecated")

from indexation import DocumentIndexer
from retrieval import DocumentRetriever
from llm import QuestionAnswerer
from evaluation import RAGEvaluator
from chatbot import RAGChatbot

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Charge la configuration depuis un fichier YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        Dictionnaire de configuration
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_config_path = os.path.join(script_dir, config_path)
    
    try:
        with open(absolute_config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"Erreur lors du chargement de la configuration: {e}")
        # Configuration par défaut pour Llama3.1
        return {
            "llm_model": "llama3.1",
            "max_length": 1024,
            "temperature": 0.7,
            "embeddings_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 1500,
            "chunk_overlap": 150,
            "persist_directory": "faiss_index",
            "data_directory": "data"
        }

def check_ollama_running() -> bool:
    """
    Vérifie si le serveur Ollama est en cours d'exécution.
    
    Returns:
        True si Ollama est en cours d'exécution, False sinon
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

def check_llama_available() -> bool:
    """
    Vérifie si le modèle Llama3.1 est disponible dans Ollama.
    
    Returns:
        True si le modèle est disponible, False sinon
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models_data = response.json().get('models', [])
            model_names = [model.get('name') for model in models_data]
            return "llama3.1" in model_names
        return False
    except Exception:
        return False

def start_ollama_server():
    """Tente de démarrer le serveur Ollama s'il n'est pas en cours d'exécution."""
    try:
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True,
                        encoding='utf-8',  # Spécifier l'encodage UTF-8
                        errors='replace')  # Remplacer les caractères non décodables
        # Attendre que le serveur démarre
        for _ in range(5):  # Attendre max 5 secondes
            time.sleep(1)
            if check_ollama_running():
                return True
        return False
    except Exception as e:
        print(f"Erreur lors du démarrage d'Ollama: {e}")
        return False

def download_llama_model():
    """
    Télécharge le modèle Llama3.1 s'il n'est pas déjà disponible.
    
    Returns:
        True si le téléchargement a réussi, False sinon
    """
    try:
        process = subprocess.Popen(
            ["ollama", "pull", "llama3.1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',  # Spécifier l'encodage UTF-8
            errors='replace'   # Remplacer les caractères non décodables
        )
        output, error = process.communicate()
        if process.returncode != 0:
            print(f"Erreur lors du téléchargement: {error}")
        return process.returncode == 0
    except Exception as e:
        print(f"Exception lors du téléchargement: {e}")
        return False

class StreamlitApp:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise l'application Streamlit.
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config
        self.indexer = None
        self.retriever = None
        self.qa = None
        self.evaluator = None
        self.chatbot = None

        # Forcer l'utilisation de Llama3.1 uniquement
        self.config["llm_model"] = "llama3.1"

        # États de session pour suivre l'indexation
        if "indexed" not in st.session_state:
            st.session_state.indexed = False
            
        # Vérifier si Ollama est en cours d'exécution
        if not check_ollama_running():
            st.warning("Le serveur Ollama n'est pas en cours d'exécution. Tentative de démarrage...")
            if start_ollama_server():
                st.success("Le serveur Ollama a été démarré avec succès!")
            else:
                st.error("Impossible de démarrer Ollama. Veuillez le démarrer manuellement avec `ollama serve`.")
                st.stop()

        # Vérifier si Llama3.1 est disponible
        if not check_llama_available():
            st.warning("Le modèle 'llama3.1' n'est pas disponible. Tentative de téléchargement...")
            with st.spinner("Téléchargement du modèle Llama3.1... Cela peut prendre plusieurs minutes."):
                if download_llama_model():
                    st.success("Le modèle Llama3.1 a été téléchargé avec succès!")
                else:
                    st.error("Échec du téléchargement de Llama3.1. Veuillez le télécharger manuellement avec `ollama pull llama3.1`.")
                    st.stop()
            
        self.initialize_components()

    def initialize_components(self):
        """
        Initialise les composants de l'application (indexeur, retriever, llm, etc.).
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_persist_directory = os.path.join(script_dir, "..", self.config["persist_directory"])

        # Initialiser l'indexeur
        self.indexer = DocumentIndexer(
            embeddings_model_name=self.config["embeddings_model"],
            persist_directory=absolute_persist_directory
        )

        # Si les documents sont déjà indexés, initialiser les autres composants
        if st.session_state.indexed:
            vectorstore = self.indexer.get_vectorstore()
            if vectorstore is not None:
                self.retriever = DocumentRetriever(vectorstore)
                
                # Utiliser uniquement le modèle Llama3.1
                self.qa = QuestionAnswerer(
                    model_name="llama3.1",
                    max_length=self.config["max_length"],
                    temperature=self.config["temperature"]
                )

                # Initialiser l'évaluateur et le chatbot si le LLM est disponible
                if self.qa.llm is not None:
                    self.evaluator = RAGEvaluator(self.qa.llm)
                    self.chatbot = RAGChatbot(vectorstore, self.qa.llm)

    def run_indexation_page(self):
        """
        Affiche la page d'indexation des documents.
        """
        st.title("Indexation des Documents")
        st.write("Cette page vous permet d'indexer vos documents PDF pour la recherche et le question-réponse.")

        with st.form("indexation_form"):
            data_dir = st.text_input("Répertoire des documents", value=self.config["data_directory"])
            
            # Paramètres optimisés
            chunk_size = st.number_input("Taille des chunks", value=self.config.get("chunk_size", 1500),
                                         help="Taille des chunks en caractères.")
            chunk_overlap = st.number_input("Chevauchement des chunks", value=self.config.get("chunk_overlap", 150),
                                           help="Chevauchement entre les chunks en caractères.")
            
            submitted = st.form_submit_button("Indexer les documents")

            if submitted:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                absolute_data_directory = os.path.join(script_dir, "..", data_dir)
                
                # Vérifier si le répertoire existe
                if not os.path.exists(absolute_data_directory):
                    st.error(f"Le répertoire {data_dir} n'existe pas. Veuillez créer ce répertoire et y placer vos documents PDF.")
                    return
                
                with st.spinner("Indexation en cours..."):
                    self.indexer.index_documents(absolute_data_directory, chunk_size, chunk_overlap)
                    
                # Vérifier si l'indexation a réussi en vérifiant la présence de l'index
                if self.indexer.get_vectorstore() is not None:
                    st.success("Indexation terminée avec succès !")
                    st.session_state.indexed = True
                    self.initialize_components()
                    st.rerun()
                else:
                    st.error("L'indexation a échoué. Veuillez vérifier vos documents et réessayer.")

    def run_search_page(self):
        """
        Affiche la page de recherche documentaire.
        """
        st.title("Recherche Documentaire")
        st.write("Recherchez des documents pertinents dans votre base de connaissances.")
        
        query = st.text_input("Votre requête de recherche")
        k = st.slider("Nombre de résultats à afficher", min_value=1, max_value=10, value=5)

        if st.button("Rechercher"):
            if query and self.retriever:
                with st.spinner("Recherche en cours..."):
                    results = self.retriever.search(query, k=k)
                    formatted_results = self.retriever.format_results(results)

                st.subheader("Résultats de la recherche")
                for i, (doc, score) in enumerate(zip(formatted_results["documents"], formatted_results["scores"])):
                    with st.expander(f"Document {i+1} (Score: {score:.4f})"):
                        st.write(doc["content"])
                        st.write("Métadonnées:", doc["metadata"])
            else:
                st.warning("Veuillez saisir une requête de recherche.")

    def run_qa_page(self):
        """
        Affiche la page de question-réponse.
        """
        st.title("Question-Answer System with Llama3.1")
        st.write("Ask natural language questions about your documents.")

        # Afficher uniquement le paramètre de température
        temperature = st.slider("Température", min_value=0.0, max_value=1.0, value=self.config.get("temperature", 0.7), step=0.1,
                              help="Valeurs plus basses pour des réponses plus déterministes, plus élevées pour plus de créativité.")
        
        # Mettre à jour le modèle si la température change
        if temperature != self.config.get("temperature"):
            self.config["temperature"] = temperature
            self.qa = QuestionAnswerer(
                model_name="llama3.1",
                max_length=self.config["max_length"],
                temperature=temperature
            )
            if self.qa.llm is not None:
                self.evaluator = RAGEvaluator(self.qa.llm)
                if self.retriever and self.retriever.vectorstore:
                    self.chatbot = RAGChatbot(self.retriever.vectorstore, self.qa.llm)

        # Interface de question-réponse
        st.subheader("Ask a Question")
        question = st.text_area("Your question", height=100, placeholder="Ask your question here...")
        k = st.slider("Number of documents to consult", min_value=1, max_value=10, value=5)

        if st.button("Answer"):
            if question and self.retriever and self.qa:
                with st.spinner("Searching for relevant documents..."):
                    results = self.retriever.search(question, k=k)
                    context_docs = [doc for doc, _ in results]

                with st.spinner("Generating answer with Llama3.1..."):
                    answer = self.qa.answer_question(question, context_docs)

                st.subheader("Answer")
                st.markdown(answer)

                # Afficher les documents utilisés
                with st.expander("Documents used for the answer"):
                    for i, (doc, score) in enumerate(results):
                        st.markdown(f"**Document {i+1}** (Score: {score:.4f})")
                        st.markdown(doc.page_content)
                        st.markdown(f"*Source: {doc.metadata.get('source', 'Not specified')}*")
                        st.markdown("---")

                # Option d'évaluation de la réponse
                if st.button("Evaluate the answer"):
                    if self.evaluator:
                        with st.spinner("Evaluation in progress..."):
                            context = "\n\n".join([doc.page_content for doc, _ in results])
                            evaluation = self.evaluator.evaluate_response(question, context, answer)

                        st.subheader("Answer Evaluation")
                        if evaluation['overall_score']:
                            st.markdown(f"**Overall score: {evaluation['overall_score']:.1f}/10**")
                        else:
                            st.markdown("**Overall score not available**")

                        col1, col2 = st.columns(2)
                        with col1:
                            with st.expander("Relevance evaluation"):
                                if evaluation['relevance']['score']:
                                    st.markdown(f"**Score: {evaluation['relevance']['score']}/10**")
                                else:
                                    st.markdown("**Score not available**")
                                st.markdown(evaluation['relevance']['evaluation'])
                        
                        with col2:
                            with st.expander("Factuality evaluation"):
                                if evaluation['factuality']['score']:
                                    st.markdown(f"**Score: {evaluation['factuality']['score']}/10**")
                                else:
                                    st.markdown("**Score not available**")
                                st.markdown(evaluation['factuality']['evaluation'])
                    else:
                        st.error("The evaluator is not available.")
            else:
                st.warning("Please enter a question.")

    def run_chatbot_page(self):
        """
        Affiche la page du chatbot RAG.
        """
        st.title("Chatbot RAG avec Llama3.1")
        st.write("Discutez avec votre base de connaissances en langage naturel.")

        # Afficher seulement la température dans la sidebar
        temperature = st.sidebar.slider("Température", min_value=0.0, max_value=1.0, value=self.config.get("temperature", 0.7), step=0.1)
        
        # Mettre à jour le modèle si la température change
        if temperature != self.config.get("temperature"):
            self.config["temperature"] = temperature
            new_qa = QuestionAnswerer(
                model_name="llama3.1",
                max_length=self.config["max_length"],
                temperature=temperature
            )
            if new_qa.llm is not None and self.retriever and self.retriever.vectorstore:
                self.chatbot = RAGChatbot(self.retriever.vectorstore, new_qa.llm)

        # Initialiser l'historique des messages s'il n'existe pas
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Afficher l'historique des messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input de chat
        if prompt := st.chat_input("Posez votre question..."):
            # Ajouter le message utilisateur à l'historique et l'afficher
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Générer et afficher la réponse
            with st.chat_message("assistant"):
                if self.chatbot:
                    with st.spinner("Réflexion en cours avec Llama3.1..."):
                        response = self.chatbot.chat(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Le chatbot n'est pas disponible. Veuillez d'abord indexer des documents.")

        # Bouton pour effacer l'historique
        if st.session_state.messages and st.sidebar.button("Effacer l'historique"):
            st.session_state.messages = []
            if self.chatbot:
                self.chatbot.clear_history()
            st.rerun()

    def run(self):
        """
        Exécute l'application principale et gère la navigation.
        """
        # Titre principal
        st.sidebar.title("Système RAG avec Llama3.1")
        
        # Navigation - suppression de l'option "Modèles Ollama"
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Page",
            ["Indexation", "Recherche", "Question-Réponse", "Chatbot"]
        )

        # Afficher la page appropriée
        if page == "Indexation":
            self.run_indexation_page()
        # Vérifier si l'indexation a été effectuée pour les autres pages
        elif st.session_state.indexed:
            if page == "Recherche":
                self.run_search_page()
            elif page == "Question-Réponse":
                self.run_qa_page()
            elif page == "Chatbot":
                self.run_chatbot_page()
        else:
            st.error("Veuillez d'abord indexer les documents dans la page 'Indexation'.")

        # Pied de page
        st.sidebar.markdown("---")
        st.sidebar.write("Projet RAG avec LangChain et Ollama")
        st.sidebar.markdown("*Optimisé pour Llama3.1*")

if __name__ == "__main__":
    # Charger la configuration
    config = load_config("../config.yaml")
    
    # Forcer l'utilisation de Llama3.1
    config["llm_model"] = "llama3.1"
    
    # Lancer l'application
    app = StreamlitApp(config)
    app.run()