from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import (ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings)

from langchain_community.vectorstores import Chroma

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model = "models/gemini-embedding-2"
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

db = Chroma(
    persist_directory=str(PROJECT_ROOT / "vector_db"),
    embedding_function=embedding,
)
retriever = db.as_retriever(
    search_kwargs={"k":3}
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)

def ask(question, history=None):

    if history is None:
        history = []

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    history_text = ""

    for message in history:
        role = message["role"]
        content = message["content"]

        if role == "user":
            history_text += f"Người dùng: {content}\n"

        elif role == "assistant":
            history_text += f"Trợ lý: {content}\n"

    prompt = f"""
    Bạn là trợ lý AI chuyên về bệnh trên cây mai vàng.

    Nhiệm vụ:
    - Trả lời dựa trên thông tin trong phần ngữ cảnh được cung cấp.
    - Ưu tiên kiến thức từ tài liệu.
    - Không tự bịa thông tin.
    - Nếu tài liệu không đủ để kết luận, hãy nói rõ rằng
    thông tin hiện có chưa đủ.
    - Trả lời bằng tiếng Việt.
    - Không chẩn đoán chắc chắn nếu chỉ có dấu hiệu không đủ rõ.

    LỊCH SỬ HỘI THOẠI:
    {history_text}

    NGỮ CẢNH TỪ CƠ SỞ TRI THỨC:
    {context}

    CÂU HỎI HIỆN TẠI:
    {question}

    CÂU TRẢ LỜI:
    """

    response = llm.invoke(prompt)

    return response.content