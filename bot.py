
import logging
import os
import time
from multiprocessing import Queue
from threading import Lock
from threading import Thread
from typing import Generator, Tuple, List
from typing import Optional

import elevenlabs
import gradio as gr
from langchain.agents import Tool, initialize_agent, AgentType, load_tools, AgentExecutor

from langchain.chains import ConversationChain, LLMMathChain, LLMChain
from langchain.chat_models import ChatOpenAI


from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import Document
from langchain.tools import (
    #PythonREPLTool,
    DuckDuckGoSearchResults,
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
)
from langchain.utilities import WikipediaAPIWrapper
from langchain.vectorstores import VectorStore

import config
from callbackhandlers import OnStream, StreamMessage
from utils._interfaces import DisposableModel
from utils.chat_tools.document_vectorstore import DocumentVectorStore
from utils.chat_tools.loader import load_faiss_vectorstore, load_chroma_vectorstore

from utils.tts import TextToSpeech

__all__ = ["MiaBot"]



class MiaBot(DisposableModel):
    _tmp_vectorstore: VectorStore
    _ttl: TextToSpeech
    _logger: logging.Logger
    chain: AgentExecutor | LLMChain

    def __init__(self, conf: config.Config, agent_type: AgentType = AgentType.OPENAI_MULTI_FUNCTIONS):
        self._logger = logging.getLogger("MiaBot")
        self._logger.debug("MiaBot initialized")
        self._ttl = TextToSpeech(os.getenv("ELEVENLABS_API_KEY"))

        self.lock = Lock()
        llm = ChatOpenAI(
            model_name=conf.openai_model_state.value,
            temperature=conf.openai_temperature_state.value,
            streaming=True,
            # max_tokens=500,
        )
        self._tmp_vectorstore = load_faiss_vectorstore()
        #self._tmp_vectorstore = load_chroma_vectorstore()

        if agent_type in [
            AgentType.OPENAI_FUNCTIONS,
            AgentType.OPENAI_MULTI_FUNCTIONS,
            AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        ]:
            agent_kwargs = {
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            }
            memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

            if agent_type in [
                AgentType.OPENAI_FUNCTIONS,
                AgentType.OPENAI_MULTI_FUNCTIONS,
            ]:
                tools = [
                    Tool(
                        name="Math",
                        func=LLMMathChain.from_llm(llm=llm, verbose=True).run,
                        description="usefull when you need to do some math",
                    ),
                    #PythonREPLTool(),
                    DuckDuckGoSearchResults(name="web_search_results"),
                    DuckDuckGoSearchRun(),
                    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
                    #VectorStoreQATool(
                    DocumentVectorStore(
                        name="uploaded_files",
                        description="useful for when you need to answer questions about an uploaded file or document. you must specify the question about the document, the filename and the source type (for example 'UPLOADED') is not mandatory",
                        vectorstore=self._tmp_vectorstore,
                    ),
                    # WolframAlphaQueryRun(),
                ]

            else:  # [ AgentType.ZERO_SHOT_REACT_DESCRIPTION, AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION ]:
                tools = load_tools(["llm-math"], llm=llm)

            self.chain = initialize_agent(
                tools=tools,
                llm=llm,
                agent=agent_type,
                memory=memory,
                agent_kwargs=agent_kwargs,
                verbose=True
            )
        else:
            self.chain = ConversationChain(llm=llm)

    def on_file_pre(self, history, file):
        self._logger.debug(f"on_file_pre file: '{file}'")
        history = history + [((file.name,), None)]
        return history

    def on_file(self, history, file):
        self._logger.debug(f"on_file file: '{file}'")
        from unstructured.partition.auto import partition
        file.seek(0)
        elements: List[str] = partition(filename=file.name)
        date = time.time()
        self._tmp_vectorstore.add_documents([
            Document(
                page_content=str(element),
                metadata={
                    "filename": os.path.basename(file.name),
                    "source": os.path.basename(file.name),
                    "source_type": "UPLOADED",
                    "page": i,
                    "time": date
                }
            )
            for i, element in enumerate(elements)
        ])
        return history

    def on_message_pre(self, history, text):
        self._logger.debug(f"on_message_pre text: '{text}'")
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)

    def _generate_response(self, question: str) -> Generator[None, str, Optional[Tuple[str, str]]]:
        def run_chain(qst: str, clb: OnStream, q: Queue):
            try:
                r = self.chain.run(qst, callbacks=[clb])
                self._logger.debug(f"response: {r}")
                if r:
                    q.put(r)
            except Exception as ex:
                self._logger.exception(ex)
                # q.put(str(ex))

        finale = Queue()
        queue = Queue()
        callback = OnStream(queue)

        thr = Thread(target=run_chain, args=(question, callback, finale))
        thr.start()
        cum = ""
        while thr.is_alive() or not queue.empty():
            try:
                message: StreamMessage = queue.get()
                if message.type == "token" and message.data:
                    cum += str(message.data)
                    yield 'partial', cum
                elif message.type == "response" and message.data:
                    yield 'final', message.data
                elif message.type == 'llm_end':
                    yield 'end', None
                else:
                    self._logger.info(f"{message.type}: invalid message - {message.data}")
            except ValueError as e:
                self._logger.exception(e)
                break
            except TypeError as e:
                self._logger.exception(e)
                break
            except EOFError as e:
                self._logger.exception(e)
                break
            except Exception as e:
                self._logger.exception(e)
                break

        response = None
        if not finale.empty():
            response = finale.get()
            yield 'complete', response

        finale.close()
        queue.close()
        return response

    def on_message(
            self,
            history,
            tts_generator_state: str = config.GENERATOR_DISABLED,
            elevenlabs_voice_id: Optional[str] = None,
            bark_voice_id: Optional[str] = None,
            bark_device: Optional[str] = config.BARK_DEVICE,
    ):
        question = history[-1][0]
        self._logger.debug(f"on_message question: '{question}' - tts_generator_state: {tts_generator_state}")
        if type(question) != str:
            yield history, None
            self._logger.debug(f"discarted question {repr(question)}")
            return

        response: Optional[str] = None
        new_msg = False
        for c, resp in self._generate_response(question):
            if c == 'partial':
                if new_msg:
                    new_msg = False
                    history.append([None, None])
                response = resp
                history[-1][1] = resp
            elif c == 'final':
                response = resp
                history[-1][1] = resp
            elif c == 'end':
                new_msg = True
            elif c == 'complete':
                response = resp
                history[-1][1] = resp
            yield history, None

        self._logger.debug(f"final response: '{response}'")

        if response is None:
            return history, None  # no response
        else:
            for r, v in self.on_message_post(
                    history,
                    tts_generator_state,
                    elevenlabs_voice_id,
                    bark_voice_id,
                    bark_device
            ):
                yield r, v

    def on_message_post(
            self,
            history,
            tts_generator_state: str = config.GENERATOR_DISABLED,
            elevenlabs_voice_id: Optional[str] = None,
            bark_voice_id: Optional[str] = None,
            bark_device: Optional[str] = config.BARK_DEVICE,
    ):
        response = history[-1][1]
        if tts_generator_state == config.GENERATOR_ELEVENLABS:
            self._logger.debug(f"generating elevenlabs audio using voice '{elevenlabs_voice_id}' ...")
            try:
                audiofile = self._ttl.elevenlabs_generate(response, elevenlabs_voice_id)
                history.append([None, (audiofile,)])
                self._logger.debug(f"audio generated: '{audiofile}'")
                yield history, audiofile
            except elevenlabs.RateLimitError as e:
                self._logger.exception(e)
                self._logger.info("TTL rate limit reached")
        elif tts_generator_state == config.GENERATOR_BARK:
            self._logger.debug(f"generating bark audio using voice '{bark_voice_id}' ...")
            audiofile = self._ttl.bark_generate(response, bark_voice_id, device=bark_device)
            history.append((None, (audiofile,)))
            self._logger.debug(f"audio generated: '{audiofile}'")
            yield history, audiofile
        else:
            self._logger.debug(f"TTL not enabled")

    def unload_model(self):
        if self._ttl:
            self._ttl.unload_model()
