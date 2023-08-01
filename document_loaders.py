from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List, Iterable
from langchain.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from unstructured.partition.auto import partition
import re


class ThreadedUnstructuredURLLoader(UnstructuredURLLoader):

    def __init__(self, urls: List[str], show_progress_bar=False, max_workers: int = 15, **kwargs):
        super().__init__(urls=urls, **kwargs)
        self._kwargs = kwargs
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self.__show_progress_bar = show_progress_bar
        self._counter = 0
        self._lock = Lock()

    def load(self) -> List[Document]:
        self._counter = 0
        docs: Iterable[List[Document]] = self._pool.map(self._load_single, self.urls)
        # if self.__show_progress_bar:
        #    docs = tqdm(docs, total=len(self.urls))
        return [item for sublist in docs for item in sublist]

    def _inc_counter(self):
        with self._lock:
            self._counter += 1
            if self.__show_progress_bar:
                print(f"Progress: {self._counter} / {len(self.urls)}", end='\r')

    def _load_single(self, url: str) -> List[Document]:
        docs = []
        try:
            # TODO: change the timeout of the request in the partition function ( in source code of unstructured -.-' )
            elements = partition(
                url=url, headers=self.headers, **self.unstructured_kwargs
            )

            if self.mode == "single":
                text = self._compress_text("\n\n".join([str(el) for el in elements]))
                metadata = {"source": url}
                docs.append(Document(page_content=text, metadata=metadata))
            elif self.mode == "elements":
                for element in elements:
                    metadata = element.metadata.to_dict()
                    metadata["category"] = element.category
                    docs.append(Document(page_content=str(element), metadata=metadata))
        except Exception as e:
            # logging.error(str(e))
            pass
        self._inc_counter()
        return docs

    def _compress_text(self, text: str):

        text = text.replace('\r\n', '\n').replace('\t', ' ')
        text = re.sub(r'[ ]+', ' ', text).replace('\n ', '\n')
        text = re.sub(r'\n+', '\n', text)
        return text
