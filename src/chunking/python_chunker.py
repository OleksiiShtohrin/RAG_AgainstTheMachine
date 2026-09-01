"""Python source code chunking strategy."""

import ast

from src.chunking.base import BaseChunker, Chunk
from src.chunking.chunk_assembler import ChunkAssembler


def _line_column_to_index(
    source: str,
    line: int,
    column: int,
) -> int:
    lines = source.splitlines(keepends=True)

    return sum(
        len(current_line)
        for current_line in lines[:line - 1]
    ) + column


def _node_to_range(
    source: str,
    node: ast.AST,
) -> tuple[int, int]:
    start = _line_column_to_index(
        source,
        node.lineno,
        node.col_offset,
    )

    end = _line_column_to_index(
        source,
        node.end_lineno,
        node.end_col_offset,
    )

    return start, end


def _find_leading_comment_start(
    content: str,
    node_start: int,
    previous_end: int,
) -> int:
    gap = content[previous_end:node_start]

    lines = gap.splitlines(keepends=True)

    position = node_start

    for line in reversed(lines):
        stripped = line.strip()

        if stripped.startswith("#"):
            position -= len(line)
        elif stripped == "":
            position -= len(line)
        else:
            break

    return position


def _is_declaration(node: ast.stmt) -> bool:
    return isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        ),
    )


class PythonChunker(BaseChunker):
    """Chunks Python source using AST logical units."""

    def __init__(
        self,
        max_chunk_size: int = 2000,
        target_chunk_size: int = 800,
        overlap: int = 150,
    ) -> None:
        """Initialize Python chunker."""
        super().__init__(
            max_chunk_size=max_chunk_size,
        )

        chunk_size = min(
            target_chunk_size,
            max_chunk_size,
        )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative"
            )

        effective_overlap = min(
            overlap,
            chunk_size - 1,
        )

        self._assembler = ChunkAssembler(
            max_chunk_size=max_chunk_size,
            target_chunk_size=target_chunk_size,
            overlap=effective_overlap,
        )

    def chunk(
        self,
        file_path: str,
        content: str,
    ) -> list[Chunk]:
        """Chunk Python source using AST logical units."""
        tree = self._parse_ast(content)

        nodes = self._get_top_level_nodes(tree)

        units = self._get_logical_units(
            content,
            nodes,
        )

        chunks: list[Chunk] = []

        for start, end in units:
            chunks.extend(
                self._assembler.assemble(
                    file_path=file_path,
                    content=content,
                    start=start,
                    end=end,
                )
            )

        return chunks

    def _parse_ast(
        self,
        content: str,
    ) -> ast.Module:
        return ast.parse(content)

    def _get_top_level_nodes(
        self,
        tree: ast.Module,
    ) -> list[ast.stmt]:
        return tree.body

    def _get_node_range(
        self,
        content: str,
        node: ast.AST,
    ) -> tuple[int, int]:
        return _node_to_range(
            content,
            node,
        )

    def _get_logical_units(
        self,
        content: str,
        nodes: list[ast.stmt],
    ) -> list[tuple[int, int]]:
        units: list[tuple[int, int]] = []

        module_start = None
        module_end = None
        previous_end = 0

        for node in nodes:
            node_start, node_end = self._get_node_range(
                content,
                node,
            )

            leading_start = _find_leading_comment_start(
                content,
                node_start,
                previous_end,
            )

            if _is_declaration(node):
                if module_start is not None:
                    units.append(
                        (
                            module_start,
                            module_end,
                        )
                    )
                    module_start = None
                    module_end = None

                units.append(
                    (
                        leading_start,
                        node_end,
                    )
                )

            else:
                if module_start is None:
                    module_start = leading_start

                module_end = node_end

            previous_end = node_end

        if module_start is not None:
            units.append(
                (
                    module_start,
                    module_end,
                )
            )

        return units
