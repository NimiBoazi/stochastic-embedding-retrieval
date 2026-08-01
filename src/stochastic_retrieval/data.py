from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import islice

from stochastic_retrieval.config import DatasetConfig


@dataclass(frozen=True)
class TextRecord:
    item_id: str
    text: str


def _field(record: object, name: str, default: str = "") -> str:
    value = getattr(record, name, default)
    return "" if value is None else str(value)


class IRDatasetAdapter:
    """Streaming adapter for ir_datasets BEIR collections."""

    def __init__(self, config: DatasetConfig) -> None:
        try:
            import ir_datasets
        except ImportError as exc:  # pragma: no cover - import message only
            raise ImportError("Install project dependencies before loading BEIR data") from exc

        self.config = config
        self.dataset = ir_datasets.load(config.ir_dataset_id)
        self._query_ids = tuple(record.item_id for record in self.iter_queries())
        self._query_id_set = set(self._query_ids)

    @property
    def query_count(self) -> int:
        return len(self._query_ids)

    @property
    def document_count(self) -> int:
        if self.config.document_limit is not None:
            return self.config.document_limit
        return int(self.dataset.docs_count())

    def iter_queries(self) -> Iterator[TextRecord]:
        records = (
            TextRecord(item_id=_field(query, "query_id"), text=_field(query, "text"))
            for query in self.dataset.queries_iter()
        )
        yield from self._limited(records, self.config.query_limit)

    def iter_documents(self) -> Iterator[TextRecord]:
        def records() -> Iterator[TextRecord]:
            for document in self.dataset.docs_iter():
                title = _field(document, "title").strip()
                body = _field(document, "text").strip()
                text = f"{title} {body}".strip()
                yield TextRecord(item_id=_field(document, "doc_id"), text=text)

        yield from self._limited(records(), self.config.document_limit)

    def qrels(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {query_id: {} for query_id in self._query_ids}
        for qrel in self.dataset.qrels_iter():
            query_id = _field(qrel, "query_id")
            if query_id in self._query_id_set:
                result[query_id][_field(qrel, "doc_id")] = int(
                    getattr(qrel, "relevance", 0)
                )
        return result

    @staticmethod
    def _limited(
        records: Iterable[TextRecord], limit: int | None
    ) -> Iterator[TextRecord]:
        yield from records if limit is None else islice(records, limit)


RecordFactory = Callable[[], Iterable[TextRecord]]
