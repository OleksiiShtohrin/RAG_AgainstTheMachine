"""Python source code chunking strategy."""

import ast

from src.chunking.base import BaseChunker, Chunk
from src.chunking.chunk_assembler import ChunkAssembler


def _line_column_to_index(
    source: str,
    line: int,
    column: int,
) -> int:
    """Convert line and column numbers into a 0-based character offset.

    Args:
        source: Complete source code string.
        line: 1-based line index.
        column: 0-based column offset.

    Returns:
        Absolute character index in the source string.
    """
    lines = source.splitlines(keepends=True)

    return sum(
        len(current_line)
        for current_line in lines[:line - 1]
    ) + column


def _node_to_range(
    source: str,
    node: ast.AST,
) -> tuple[int, int]:
    """Calculate character index start and end boundaries for an AST node.

    Args:
        source: Complete source code string.
        node: Target AST node with line and column positions.

    Returns:
        Tuple of (start_char_index, end_char_index).

    Raises:
        ValueError: If node lacks line or column positioning attributes.
    """
    start_line = getattr(node, "lineno", None)
    start_column = getattr(node, "col_offset", None)
    end_line = getattr(node, "end_lineno", None)
    end_column = getattr(node, "end_col_offset", None)

    if (
        not isinstance(start_line, int)
        or not isinstance(start_column, int)
        or not isinstance(end_line, int)
        or not isinstance(end_column, int)
    ):
        raise ValueError("AST node has no source position")

    start = _line_column_to_index(
        source,
        start_line,
        start_column,
    )

    end = _line_column_to_index(
        source,
        end_line,
        end_column,
    )

    return start, end


def _find_leading_comment_start(
    content: str,
    node_start: int,
    previous_end: int,
) -> int:
    """Find start offset of comments immediately preceding an AST node.

    Args:
        content: Complete source code string.
        node_start: Character offset where the AST node begins.
        previous_end: Ending offset of the preceding AST node.

    Returns:
        Character offset where leading comments begin.
    """
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
    """Check whether an AST statement is a function or class definition.

    Args:
        node: Statement AST node to inspect.

    Returns:
        True if the statement is a class or function declaration.
    """
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
        """Initialize Python chunker with size and overlap limits.

        Args:
            max_chunk_size: Maximum characters allowed per chunk.
            target_chunk_size: Target slice size.
            overlap: Character overlap between consecutive chunks.

        Raises:
            ValueError: If overlap is negative.
        """
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
        """Chunk Python source using AST logical units with fallback.

        Args:
            file_path: Relative path to the Python source file.
            content: Complete Python file content.

        Returns:
            List of generated Chunk instances.
        """
        if not content.strip():
            return []

        try:
            tree = self._parse_ast(content)
            nodes = self._get_top_level_nodes(tree)
            units = self._get_logical_units(content, nodes)

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
        except (SyntaxError, ValueError):
            # Fallback if AST parsing fails on broken or non-standard syntax
            return self._assembler.assemble(
                file_path=file_path,
                content=content,
                start=0,
                end=len(content),
            )

    def _parse_ast(
        self,
        content: str,
    ) -> ast.Module:
        """Parse source content string into an AST module.

        Args:
            content: Raw Python code.

        Returns:
            Parsed ast.Module node.
        """
        return ast.parse(content)

    def _get_top_level_nodes(
        self,
        tree: ast.Module,
    ) -> list[ast.stmt]:
        """Extract top-level statement nodes from an AST module.

        Args:
            tree: Root ast.Module node.

        Returns:
            List of top-level statement nodes.
        """
        return tree.body

    def _get_node_range(
        self,
        content: str,
        node: ast.AST,
    ) -> tuple[int, int]:
        """Get start and end character offsets for an AST node.

        Args:
            content: Complete source code string.
            node: AST node.

        Returns:
            Tuple of (start_char_index, end_char_index).
        """
        return _node_to_range(
            content,
            node,
        )

    def _get_logical_units(
        self,
        content: str,
        nodes: list[ast.stmt],
    ) -> list[tuple[int, int]]:
        """Group statements into logical spans (functions, classes, imports).

        Args:
            content: Complete source code string.
            nodes: Top-level statement nodes.

        Returns:
            List of (start, end) character tuples representing logical units.
        """
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
                if module_start is not None and module_end is not None:
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

        if module_start is not None and module_end is not None:
            units.append(
                (
                    module_start,
                    module_end,
                )
            )

        return units
