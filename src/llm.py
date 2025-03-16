# llm.py (Version optimisée pour llama3)
from typing import List, Any, Dict
import warnings
from langchain_community.llms import Ollama  # Utilisation de l'ancienne classe
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain

# Supprimer les avertissements de dépréciation pour cette classe
warnings.filterwarnings("ignore", message="The class `Ollama` was deprecated")
warnings.filterwarnings("ignore", message="This class is deprecated. See the following migration guides")

class QuestionAnswerer:
    def __init__(self, model_name: str = "llama3", max_length: int = 1024, temperature: float = 0.7):
        """
        Initialise le système de question-réponse avec llama3.

        Args:
            model_name: Nom du modèle Ollama à utiliser (forcé à llama3)
            max_length: Longueur maximale des réponses générées
            temperature: Température pour la génération de texte (0.0-1.0)
        """
        self.llm = None
        self.prompt = None
        self.qa_chain = None
        
        # On force l'utilisation de llama3
        model_name = "llama3"
        
        # Template de prompt optimisé pour llama3
        self.template = """
            Tu es un assistant IA précis qui répond aux questions en se basant uniquement sur le contexte fourni.
            
            Contexte: {context}
            
            Question: {question}
            
            Répondez à la question en vous basant UNIQUEMENT sur le contexte ci-dessus.
            Si le contexte ne contient pas l'information nécessaire, indiquez-le clairement.
            Soyez précis et factuel.
            
            Réponse:
            """

        try:
            print(f"Connexion au serveur Ollama avec le modèle llama3...")
            
            # Initialiser le LLM Ollama avec des paramètres optimisés pour llama3
            self.llm = Ollama(
                model="llama3",
                temperature=temperature,
                num_ctx=4096,      # Contexte plus large 
                num_predict=max_length,
                top_p=0.95,        # Plus restrictif pour plus de précision
                top_k=40,          # Paramètre supplémentaire pour Llama
                repeat_penalty=1.1
            )
            
            # Configuration du prompt
            self.prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=self.template
            )
            
            # Créer une simple LLMChain au lieu de load_qa_chain
            self.qa_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt
            )
            
            print(f"Modèle Ollama llama3 connecté avec succès!")

        except Exception as e:
            print(f"Erreur lors de la connexion à Ollama: {e}")
            print("Assurez-vous que le serveur Ollama est en cours d'exécution (ollama serve) et que le modèle llama3 est disponible.")
            print("Vous pouvez télécharger le modèle avec 'ollama pull llama3'")
            
            # Pas de fallback - si Ollama ne fonctionne pas, on doit le corriger
            self.llm = None
            self.prompt = None
            self.qa_chain = None

    def answer_question(self, question: str, context_docs: List[Any]) -> str:
        try:
            if self.qa_chain is None:
                return "Désolé, la connexion à Ollama n'a pas pu être établie. Vérifiez que le serveur est en cours d'exécution et que le modèle llama3 est disponible."
                
            # Formater le contexte à partir des documents
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Utiliser invoke au lieu de run (méthode moderne)
            answer = self.qa_chain.invoke({
                "context": context,
                "question": question
            })
            
            return answer["text"] if isinstance(answer, dict) and "text" in answer else answer
            
        except Exception as e:
            print(f"Erreur lors de la génération de la réponse: {e}")
            return "Désolé, je n'ai pas pu générer une réponse valide. Veuillez réessayer."