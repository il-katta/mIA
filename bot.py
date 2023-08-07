import logging
import os
from multiprocessing import Queue
from threading import Lock
from threading import Thread
from typing import List, Generator, Tuple
from typing import Optional

import elevenlabs
import gradio as gr

from langchain.chains import ConversationChain, LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType, load_tools

import config
from callbackhandlers import OnStream, StreamMessage
from utils.tts import TextToSpeech

__all__ = ["MiaBot"]


class MiaBot:
    def __init__(self, conf: config.Config, agent_type: AgentType = AgentType.OPENAI_MULTI_FUNCTIONS):
        self.lock = Lock()
        llm = ChatOpenAI(
            model_name=conf.openai_model_state.value,
            temperature=conf.openai_temperature_state.value,
            streaming=True,
            # max_tokens=500,
        )

        if agent_type == AgentType.OPENAI_FUNCTIONS or agent_type == AgentType.OPENAI_MULTI_FUNCTIONS:
            tools = [
                Tool(
                    name="Math",
                    func=LLMMathChain.from_llm(llm=llm, verbose=True).run,
                    description="usefull when you need to do some math",
                ),
            ]
            self.chain = initialize_agent(llm=llm, tools=tools, agent=agent_type, verbose=True)
        elif agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
            tools = load_tools(["llm-math"], llm=llm)
            self.chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        else:
            self.chain = ConversationChain(llm=llm)

        self._logger = logging.getLogger("MiaBot")
        self._logger.debug("MiaBot initialized")
        self._ttl = TextToSpeech(os.getenv("ELEVENLABS_API_KEY"))

    def on_file_pre(self, history, file):
        self._logger.debug(f"on_file_pre file: '{file}'")
        history = history + [((file.name,), None)]
        return history

    def on_message_pre(self, history, text):
        self._logger.debug(f"on_message_pre text: '{text}'")
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)

    def _generate_response(self, question: str) -> Generator[None, str, Optional[Tuple[str,str]]]:
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
            bark_voice_id: Optional[str] = None
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
            audiofile = self._ttl.bark_generate(response, bark_voice_id)
            history.append((None, (audiofile,)))
            self._logger.debug(f"audio generated: '{audiofile}'")
            yield history, audiofile
        else:
            self._logger.debug(f"TTL not enabled")


