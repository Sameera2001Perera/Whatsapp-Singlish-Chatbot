# from transliterator.forward import ForwardTransliterator

# forward_transliterator = ForwardTransliterator()
# aa = forward_transliterator.transliterate("ශ්රී ලංකාවේ භූගෝල විද්යාව හෝ ස්ථානය ගැන ඔබ දැන ගැනීමට කැමති වෙනත් යමක් තිබේද?")
# print(aa)

from transliterator.transliteration import Transliterator
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


model_directory = "Ransaka/sinhala-bert-medium-v2"
dictionary_path = "data/dictionary.txt"
tokenizer = AutoTokenizer.from_pretrained(model_directory)
model = AutoModelForMaskedLM.from_pretrained(model_directory)
model.eval()

transliterator = Transliterator(
    dictionary_path=dictionary_path, tokenizer=tokenizer, model=model
)

sinhala_text = transliterator.generate_sinhala("mge rata dnnwd")
print(sinhala_text)