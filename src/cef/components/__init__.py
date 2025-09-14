from __future__ import annotations
from typing import Dict, Type

from .selectors import InstructionKeywordSelector, FactKeywordSelector, FactEmbeddingSelector, ToolKeywordSelector
from .compressors import DecisionSummarizer
from .writers import ScratchpadFileWriter


COMPONENTS: Dict[str, Type] = {
    # selectors
    "instruction.keyword": InstructionKeywordSelector,
    "fact.keyword": FactKeywordSelector,
    "fact.embedding": FactEmbeddingSelector,
    "tool.rag": ToolKeywordSelector,
    # compressors
    "summarize.decision": DecisionSummarizer,
    # writers
    "scratchpad.file": ScratchpadFileWriter,
}

