import logging
from flask import current_app, jsonify
import json
import requests
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transliterator.transliteration import Transliterator
from easygoogletranslate import EasyGoogleTranslate
from transliterator.forward import ForwardTransliterator
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, trim_messages
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages.utils import count_tokens_approximately
import os
from langchain_anthropic import ChatAnthropic
from ..doc_reader import DocumentReader


def delete_file(file_path):
    """
    Delete a file at the specified path.

    Args:
        file_path (str): The path to the file to be deleted

    Returns:
        bool: True if file was successfully deleted, False otherwise
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logging.warning(f"File does not exist: {file_path}")
            return False

        # Check if it's actually a file (not a directory)
        if not os.path.isfile(file_path):
            logging.error(f"Path is not a file: {file_path}")
            return False

        # Delete the file
        os.remove(file_path)
        logging.info(f"Successfully deleted file: {file_path}")
        return True

    except PermissionError:
        logging.error(f"Permission denied: Cannot delete file {file_path}")
        return False
    except OSError as e:
        logging.error(f"OS error occurred while deleting file {file_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error while deleting file {file_path}: {e}")
        return False


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


trimmer = trim_messages(
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=1024,
    start_on="human",
    end_on=("human"),
    include_system=True,
    allow_partial=False,
)


sys_message = """
You are a friendly, helpful, and professional chatbot interacting with users on WhatsApp. Always greet users politely and respond in a clear, concise, and conversational tone.

Users send messages in Romanized Sinhala (commonly known as Singlish), which is Sinhala written using the Roman alphabet. To type faster, users often omit vowels or follow unconventional spelling patterns.

Each message is processed through a transliteration system that converts Singlish into native Sinhala script, accounting for common typing variations such as vowel omissions. The resulting Sinhala text is then translated into English for your understanding.

For every user query, you will receive three versions: the original Singlish input, the transliterated Sinhala text, and the English translation.

Be aware that both transliteration and translation can introduce errors. Always consider all three inputs before forming your response.

Respond in English.
"""

rag_sys_message = """
You are a friendly, helpful, and professional chatbot interacting with users on WhatsApp. Always greet users politely and respond in a clear, concise, and conversational tone.

Users send messages in Romanized Sinhala (commonly known as Singlish), which is Sinhala written using the Roman alphabet. To type faster, users often omit vowels or follow unconventional spelling patterns.

Each message is processed through a transliteration system that converts Singlish into native Sinhala script, accounting for common typing variations such as vowel omissions. The resulting Sinhala text is then translated into English for your understanding.

For every user query, you will receive three versions: the original Singlish input, the transliterated Sinhala text, and the English translation.

Be aware that both transliteration and translation can introduce errors. Always consider all three inputs before forming your response.

Respond in English. For some queries, you may need to retrieve relevant information from a knowledge base. Below is the information you can use to answer user queries.

### Knowledge from the uploaded document:
{knowledge_base}
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", sys_message),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Singlish Query: {singlish}\n Transliterated Sinhala: {sinhala}\nEnglish Translation: {english}"),
])

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_sys_message),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Singlish Query: {singlish}\n Transliterated Sinhala: {sinhala}\nEnglish Translation: {english}"),
])

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0,
    max_tokens=8192,
    timeout=None,
    max_retries=2,
    # other params...
)

chain = prompt | trimmer | llm
# rag_chain = rag_prompt | trimmer | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="english",
    history_messages_key="history",
)

# rag_chain_with_history = RunnableWithMessageHistory(
#     rag_chain,
#     get_by_session_id,
#     input_messages_key="english",
#     history_messages_key="history",
# )


model_directory = "Ransaka/sinhala-bert-medium-v2"
dictionary_path = "data/dictionary.txt"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForMaskedLM.from_pretrained(model_directory)
model.eval()
transliterator = Transliterator(
    dictionary_path=dictionary_path, tokenizer=tokenizer, model=model
)
forward_transliterator = ForwardTransliterator()
translator_en_si = EasyGoogleTranslate(
    source_language='en',
    target_language='si',
    timeout=20
)
translator_si_en = EasyGoogleTranslate(
    source_language='si',
    target_language='en',
    timeout=20
)

doc_reader = DocumentReader()



def log_http_response(response):
    logging.info(f"Status: {response.status_code}")
    logging.info(f"Content-type: {response.headers.get('content-type')}")
    logging.info(f"Body: {response.text}")


def get_text_message_input(recipient, text):
    return json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }
    )


def generate_response(response):
    # Return text
    return response


def send_message(data):
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {current_app.config['ACCESS_TOKEN']}",
    }

    url = f"https://graph.facebook.com/{current_app.config['VERSION']}/{current_app.config['PHONE_NUMBER_ID']}/messages"

    try:
        response = requests.post(
            url, data=data, headers=headers, timeout=10
        )  # 10 seconds timeout as an example
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.Timeout:
        logging.error("Timeout occurred while sending message")
        return jsonify({"status": "error", "message": "Request timed out"}), 408
    except (
        requests.RequestException
    ) as e:  # This will catch any general request exception
        logging.error(f"Request failed due to: {e}")
        return jsonify({"status": "error", "message": "Failed to send message"}), 500
    else:
        # Process the response as normal
        log_http_response(response)
        return response


def download_media(media_id, media_type, wa_id=None, filename_hint=None):
    """
    Downloads media by ID and saves it with correct extension if available.
    """
    access_token = current_app.config['ACCESS_TOKEN']

    try:
        # Step 1: Get download URL
        media_url_resp = requests.get(
            f"https://graph.facebook.com/v19.0/{media_id}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        media_url_resp.raise_for_status()
        media_url = media_url_resp.json()["url"]

        # Step 2: Download the media content
        media_data_resp = requests.get(
            media_url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        media_data_resp.raise_for_status()

        # Step 3: Determine save path
        folder = f"media/{wa_id}" if wa_id else "media"
        os.makedirs(folder, exist_ok=True)

        if media_type == "document" and filename_hint:
            filename = filename_hint
        else:
            # Fallback filename
            filename = f"{media_type}_{media_id}.bin"

        filepath = os.path.join(folder, filename)

        with open(filepath, "wb") as f:
            f.write(media_data_resp.content)

        documents = doc_reader.load_and_split(filepath, wa_id)
        doc_reader.add_documents(documents)

        delete_file(filepath)  # Clean up the file after processing

        logging.info(f"Saved {media_type} to {filepath}")
        return filepath

    except Exception as e:
        logging.exception(f"Failed to download media: {e}")
        return None


# def process_whatsapp_message(body):
#     wa_id = body["entry"][0]["changes"][0]["value"]["contacts"][0]["wa_id"]
#     name = body["entry"][0]["changes"][0]["value"]["contacts"][0]["profile"]["name"]

#     print(f"\n\nReceived message from {name} ({wa_id})")

#     message = body["entry"][0]["changes"][0]["value"]["messages"][0]
#     message_body = message["text"]["body"]
#     sinhala_text = transliterator.generate_sinhala(message_body)
#     english_text = translator_si_en.translate(sinhala_text)

#     # TODO: Add logic to generate a response based on the English text
#     english_response = chain_with_history.invoke(  # noqa: T201
#         {"singlish": message_body, "sinhala": sinhala_text, "english": english_text},
#         config={"configurable": {"session_id": wa_id}}
#         )

#     sinhala_response = translator_en_si.translate(english_response.content)
#     singlish_response = forward_transliterator.transliterate(sinhala_response)

#     print(f"\nSinglish Input: {message_body}\n\n")
#     print(f"Transliterated Sinhala: {sinhala_text}\n\n")
#     print(f"English Translation: {english_text}\n\n")
#     print(f"English Response: {english_response.content}\n\n")
#     print(f"Transliterated Sinhala Response: {sinhala_response}\n\n")



#     response = generate_response(singlish_response)

#     data = get_text_message_input(current_app.config["RECIPIENT_WAID"], response)
#     send_message(data)


def process_whatsapp_message(body):
    wa_id = body["entry"][0]["changes"][0]["value"]["contacts"][0]["wa_id"]
    name = body["entry"][0]["changes"][0]["value"]["contacts"][0]["profile"]["name"]

    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    msg_type = message["type"]

    if msg_type == "text":
        message_body = message["text"]["body"]

        print("AAAAAAAAAAAAAAAAAAAAAAAAAAA")

        if message_body[0]=='#':
            print("BBBBBBBBBBBBBBBBBBBBBBBBBB")
            # sinhala_text = transliterator.generate_sinhala(message_body[1:])
            # english_text = translator_si_en.translate(sinhala_text)
            # info = doc_reader.retrieve(english_text, wa_id, k=2)
            # english_response = rag_chain_with_history.invoke(
            #     {"knowledge_base": info, "singlish": message_body, "sinhala": sinhala_text, "english": english_text},
            #     config={"configurable": {"session_id": wa_id}}
            # )
            # sinhala_response = translator_en_si.translate(english_response.content)
            # singlish_response = forward_transliterator.transliterate(sinhala_response)
        elif message_body.startswith("/new"):
            # delete history and vector store for this user with wa_id
            store.pop(wa_id, None)  # Clear chat history
            doc_reader.delete_documents(wa_id)
            print(f"store: {store}")
        else:
            sinhala_text = transliterator.generate_sinhala(message_body)
            english_text = translator_si_en.translate(sinhala_text)
            english_response = chain_with_history.invoke(
                {"singlish": message_body, "sinhala": sinhala_text, "english": english_text},
                config={"configurable": {"session_id": wa_id}}
            )
            sinhala_response = translator_en_si.translate(english_response.content)
            singlish_response = forward_transliterator.transliterate(sinhala_response)


        response = generate_response(singlish_response)
        data = get_text_message_input(wa_id, response)
        send_message(data)

    elif msg_type in ["image", "video", "audio", "document"]:
        media_id = message[msg_type]["id"]
        caption = message[msg_type].get("caption", "")

        # For documents only, try to get original filename
        filename_hint = message[msg_type].get("filename") if msg_type == "document" else None

        saved_path = download_media(
            media_id=media_id,
            media_type=msg_type,
            wa_id=wa_id,
            filename_hint=filename_hint,
        )

        if saved_path:
            ack_text = f"Thanks for the {msg_type}!"
        else:
            ack_text = f"Could not save your {msg_type}, please try again."

        data = get_text_message_input(wa_id, ack_text)
        send_message(data)
    else:
        print(f"Unhandled message type: {msg_type}")


def is_valid_whatsapp_message(body):
    """
    Check if the incoming webhook event has a valid WhatsApp message structure.
    """
    return (
        body.get("object")
        and body.get("entry")
        and body["entry"][0].get("changes")
        and body["entry"][0]["changes"][0].get("value")
        and body["entry"][0]["changes"][0]["value"].get("messages")
        and body["entry"][0]["changes"][0]["value"]["messages"][0]
    )
