# chatbot.py (Version optimisée pour Llama3.1)
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS

class RAGChatbot:
    def __init__(self, vectorstore: FAISS, llm: OllamaLLM):
        self.vectorstore = vectorstore
        self.llm = llm
        self.chat_history = ChatMessageHistory()

        # Créer un prompt template qui inclut l'historique de chat
        # Optimisé pour Llama 3.1 avec des instructions plus précises pour la reformulation
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a competent AI assistant who answers questions based on the information provided in the context.
            
            Important guidelines:
            1. Always respond in English, regardless of the language of the question
            2. Reformulate the information in your own words rather than copy-pasting the raw content
            3. Use clear and concise language
            4. Stay factual and base your answers only on the provided content
            5. If the question is unclear, ask for clarification
            6. Structure your response logically
            7. If you cannot find the information in the context, state it clearly
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
            ("system", "Here is the relevant context to answer the question: {context}")
        ])

        # Créer la chaîne de récupération et de réponse avec l'API LCEL moderne
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Initial k=3

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough(), "chat_history": lambda _: self.chat_history.messages}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Fallback chain with larger k:
        self.fallback_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        self.fallback_chain = (
             {"context": self.fallback_retriever | format_docs, "question": RunnablePassthrough(), "chat_history": lambda _: self.chat_history.messages}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
    def _extract_titles(self, docs: List[Any]) -> str:
        """
        Tente d'extraire uniquement les titres et les informations structurées.
        Pour éviter de simplement copier le contenu brut.
        """
        # Ne pas extraire directement le contenu - retourner une chaîne vide
        # pour forcer l'utilisation du LLM pour formuler une réponse cohérente
        return ""

    def chat(self, message: str) -> str:
        self.chat_history.add_message(HumanMessage(content=message))

        # Récupérer les documents pertinents
        initial_docs = self.retriever.get_relevant_documents(message)
        
        # Utiliser directement le LLM pour générer une réponse
        response = self.rag_chain.invoke(message)

        # Check if the response indicates no information:
        if "ne contient pas l'information nécessaire" in response.lower() or "je n'ai pas pu générer une réponse valide" in response.lower():
            # Try fallback with larger k:
            fallback_response = self.fallback_chain.invoke(message)
            if "ne contient pas l'information nécessaire" in fallback_response.lower() or "je n'ai pas pu générer une réponse valide" in fallback_response.lower():
                # Still no answer:
                self.chat_history.add_message(AIMessage(content="I couldn't find any information on this topic in the available documents. Please make sure the appropriate documents are indexed."))
                return "I couldn't find any information on this topic in the available documents. Please make sure the appropriate documents are indexed."
            else:
                self.chat_history.add_message(AIMessage(content=fallback_response))
                return fallback_response
        else:
             self.chat_history.add_message(AIMessage(content=response))
             return response

    def get_history(self) -> List[Dict[str, str]]:
        history = []
        for message in self.chat_history.messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            history.append({
                "role": role,
                "content": message.content
            })
        return history
        
    def clear_history(self) -> None:
        self.chat_history.clear()