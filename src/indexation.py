# indexation.py
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from typing import List, Any, Optional

class DocumentIndexer:
    def __init__(self, embeddings_model_name: str, persist_directory: str):
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        self.persist_directory = persist_directory
        self.vectorstore = None

    def load_documents(self, directory: str) -> List[Any]:
        """
        Charge les documents PDF et TXT depuis un répertoire.
        """
        try:
            print(f"Chargement des documents depuis {directory}...")
            documents = []
            # Load PDFs
            pdf_loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents.extend(pdf_loader.load())
            # Load TXT files (for simple course lists, etc.)
            txt_loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
            documents.extend(txt_loader.load())

            if not documents:
                print("Aucun document PDF ou TXT trouvé.")
                return []

            print(f"{len(documents)} documents chargés avec succès.")
            return documents
        except Exception as e:
            print(f"Erreur lors du chargement des documents: {e}")
            return []

    def split_documents(self, documents: List[Any], chunk_size: int = 1500, chunk_overlap: int = 150) -> List[Any]:
        # ... (rest of the split_documents method remains the same) ...
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Documents divisés en {len(chunks)} chunks.")
        return chunks


    def create_embeddings(self, chunks: List[Any]) -> None:
        # ... (rest of the create_embeddings method remains the same)
        print("Création des embeddings et de l'index vectoriel...")
        self.vectorstore = FAISS.from_documents(documents=chunks, embedding=self.embeddings)

        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        index_path = os.path.join(self.persist_directory, "index") # Use persist_dir directly
        self.vectorstore.save_local(index_path)
        print(f"Index vectoriel sauvegardé dans {index_path}")

    def index_documents(self, directory: str, chunk_size: int, chunk_overlap: int) -> None:
        # ... (rest of index_documents method is the same)
        documents = self.load_documents(directory)
        if documents:
            chunks = self.split_documents(documents, chunk_size, chunk_overlap)
            if chunks:
                self.create_embeddings(chunks)
                print("Indexation terminée avec succès!")
            else:
                print("Aucun chunk créé. Vérifiez le contenu des documents.")
        else:
            print("Aucun document trouvé. Vérifiez le chemin du répertoire.")


    def get_vectorstore(self) -> Optional[Any]:
        # ... (rest of get_vectorstore method is the same)
        if self.vectorstore is None:
            index_path = os.path.join(self.persist_directory, "index") # Use persist_dir
            if os.path.exists(index_path):
                print(f"Chargement de l'index vectoriel depuis {index_path}...")
                try:
                    self.vectorstore = FAISS.load_local(
                        index_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True  # Only when loading
                    )
                    print("Index vectoriel chargé avec succès!")
                except Exception as e:
                    print(f"Erreur lors du chargement de l'index: {e}")
                    return None
            else:
                print(f"Aucun index trouvé à {index_path}. Veuillez d'abord indexer les documents.")
                return None
        return self.vectorstore