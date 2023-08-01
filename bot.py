import logging
import os
from multiprocessing import Queue
from threading import Lock
from threading import Thread
from typing import List
from typing import Optional

import elevenlabs
import gradio as gr
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

import config
from callbackhandlers import OnStream, StreamMessage
from ttl import TextToVoice

__all__ = ["MiaBot"]


class MiaBot:
    def __init__(self, conf: config.Config):
        self.lock = Lock()
        llm = ChatOpenAI(
            model_name=conf.openai_model_state.value,
            temperature=conf.openai_temperature_state.value,
            streaming=True,
            #max_tokens=500,
        )
        self.chain = ConversationChain(llm=llm)
        self._logger = logging.getLogger("MiaBot")
        self._logger.debug("MiaBot initialized")
        self._ttl = TextToVoice(os.getenv("ELEVENLABS_API_KEY"))

    def on_file_pre(self, history, file):
        self._logger.debug(f"on_file_pre file: '{file}'")
        history = history + [((file.name,), None)]
        return history

    def on_message_pre(self, history, text):
        self._logger.debug(f"on_message_pre text: '{text}'")
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)

    def _generate_response(self, question: str, history: List[List[str]], on_final_response):
        def run_chain(qst: str, clb: OnStream, q: Queue):
            r = self.chain.run(qst, callbacks=[clb])
            self._logger.debug(f"response: {r}")
            if r:
                q.put(r)

        finale = Queue()
        queue = Queue()
        callback = OnStream(queue)

        thr = Thread(target=run_chain, args=(question, callback, finale))
        thr.start()
        while thr.is_alive() or not queue.empty():
            try:
                message: StreamMessage = queue.get()
                if message.type == "token" and message.data:
                    token = message.data
                    history[-1][1] += str(token)
                    yield history, None
                if message.type == "response" and message.data:
                    history[-1][1] = message.data
                    yield history, None
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

        if not finale.empty():
            response = finale.get()
        else:
            response = history[-1][1]
        finale.close()
        queue.close()
        on_final_response(response)

    def on_message(
            self,
            history,
            ttl_generator_state: str = config.GENERATOR_DISABLED,
            elevenlabs_voice_id: Optional[str] = None,
            bark_voice_id: Optional[str] = None
    ):
        question = history[-1][0]
        self._logger.debug(f"on_message question: '{question}' - ttl_generator_state: {ttl_generator_state}")
        if type(question) != str:
            yield history, None
            self._logger.debug(f"discarted question {repr(question)}")
            return

        history[-1][1] = ""

        response = None

        def on_final_response(rsp):
            global response
            response = rsp

        for a, b in self._generate_response(question, history, on_final_response):
            yield a, b

        if response is None:
            response = history[-1][1]

        self._logger.debug(f"final response: '{response}'")
        if ttl_generator_state == config.GENERATOR_ELEVENLABS:
            self._logger.debug(f"generating elevenlabs audio using voice '{elevenlabs_voice_id}' ...")
            try:
                audiofile = self._ttl.elevenlabs_generate(response, elevenlabs_voice_id)
                history.append((None, (audiofile,)))
                self._logger.debug(f"audio generated: '{audiofile}'")
                yield history, audiofile
            except elevenlabs.RateLimitError as e:
                self._logger.exception(e)
                self._logger.info("TTL rate limit reached")
        elif ttl_generator_state == config.GENERATOR_BARK:
            self._logger.debug(f"generating bark audio using voice '{bark_voice_id}' ...")
            audiofile = self._ttl.bark_generate(response, bark_voice_id)
            history.append((None, (audiofile,)))
            self._logger.debug(f"audio generated: '{audiofile}'")
            yield history, audiofile
        else:
            self._logger.debug(f"TTL not enabled")
