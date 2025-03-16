# --- evaluation.py mis à jour pour utiliser les API modernes de LangChain avec Ollama ---
from typing import List, Dict, Any
from langchain_ollama import OllamaLLM  # Utiliser le nouveau package
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAGEvaluator:
    """
    Classe responsable de l'évaluation des réponses générées par le système RAG.
    Mise à jour pour utiliser Ollama avec Llama 3.2 et l'API moderne de LangChain.
    """

    def __init__(self, llm: OllamaLLM):
        """
        Initialise l'évaluateur avec un LLM Ollama.

        Args:
            llm: Modèle de langage Ollama à utiliser pour l'évaluation
        """
        self.llm = llm

        # Template de prompt pour l'évaluation de la pertinence, optimisé pour Llama 3.2
        self.relevance_template = """
        En tant qu'évaluateur objectif, votre tâche est d'analyser la pertinence d'une réponse:

        Question: {question}
        Contexte: {context}
        Réponse générée: {answer}

        Évaluez la pertinence de la réponse par rapport à la question et au contexte.
        Attribuez un score de 1 à 10 (10 étant excellent) et justifiez votre évaluation.
        Soyez précis et factuel dans votre analyse.

        Score (1-10):
        """

        self.relevance_prompt = PromptTemplate(
            input_variables=["question", "context", "answer"],
            template=self.relevance_template
        )

        # Utilisation de l'API LCEL moderne
        self.relevance_chain = (
            self.relevance_prompt 
            | self.llm 
            | StrOutputParser()
        )

        # Template de prompt pour l'évaluation de la factualité, optimisé pour Llama 3.2
        self.factuality_template = """
        En tant qu'évaluateur objectif, votre tâche est d'analyser la factualité d'une réponse:

        Contexte: {context}
        Réponse générée: {answer}

        Évaluez si la réponse est fidèle aux faits présentés dans le contexte.
        Y a-t-il des hallucinations ou des informations non présentes dans le contexte?
        Attribuez un score de 1 à 10 (10 étant excellent) et justifiez votre évaluation.
        Soyez précis et factuel dans votre analyse.

        Score (1-10):
        """

        self.factuality_prompt = PromptTemplate(
            input_variables=["context", "answer"],
            template=self.factuality_template
        )

        # Utilisation de l'API LCEL moderne
        self.factuality_chain = (
            self.factuality_prompt 
            | self.llm 
            | StrOutputParser()
        )

    def evaluate_relevance(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """
        Évalue la pertinence de la réponse.

        Args:
            question: La question posée
            context: Le contexte utilisé
            answer: La réponse générée

        Returns:
            Dictionnaire contenant le score et la justification
        """
        # Utiliser invoke au lieu de run/call
        evaluation = self.relevance_chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        })

        # Extraction du score
        try:
            score_line = [line for line in evaluation.split('\n') if 'Score' in line][0]
            score = int(score_line.split(':')[1].strip())
        except:
            score = None

        return {
            "score": score,
            "evaluation": evaluation
        }

    def evaluate_factuality(self, context: str, answer: str) -> Dict[str, Any]:
        """
        Évalue la factualité de la réponse.

        Args:
            context: Le contexte utilisé
            answer: La réponse générée

        Returns:
            Dictionnaire contenant le score et la justification
        """
        # Utiliser invoke au lieu de run/call
        evaluation = self.factuality_chain.invoke({
            "context": context,
            "answer": answer
        })

        # Extraction du score
        try:
            score_line = [line for line in evaluation.split('\n') if 'Score' in line][0]
            score = int(score_line.split(':')[1].strip())
        except:
            score = None

        return {
            "score": score,
            "evaluation": evaluation
        }

    def evaluate_response(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """
        Évalue complètement une réponse.

        Args:
            question: La question posée
            context: Le contexte utilisé
            answer: La réponse générée

        Returns:
            Dictionnaire contenant les scores et évaluations
        """
        relevance = self.evaluate_relevance(question, context, answer)
        factuality = self.evaluate_factuality(context, answer)

        # Calculer un score global
        if relevance["score"] is not None and factuality["score"] is not None:
            overall_score = (relevance["score"] + factuality["score"]) / 2
        else:
            overall_score = None

        return {
            "relevance": relevance,
            "factuality": factuality,
            "overall_score": overall_score
        }