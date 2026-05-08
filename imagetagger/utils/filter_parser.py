from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from imagetagger.ui.main_window import ImageRecord


class FilterSyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class _FilterToken:
    kind: str
    value: str
    position: int


class _FilterNode:
    def evaluate(self, record: "ImageRecord", runtime: "_FilterRuntime") -> bool:
        raise NotImplementedError()


@dataclass(frozen=True)
class _NamedFilterNode(_FilterNode):
    name: str

    def evaluate(self, record: "ImageRecord", runtime: "_FilterRuntime") -> bool:
        predicate = runtime.named_filters.get(self.name.casefold())
        if predicate is None:
            return False
        return predicate(record)


@dataclass(frozen=True)
class _TagFilterNode(_FilterNode):
    tag: str

    def evaluate(self, record: "ImageRecord", runtime: "_FilterRuntime") -> bool:
        return runtime.tag_filter(record, self.tag)


@dataclass(frozen=True)
class _FreetextFilterNode(_FilterNode):
    text: str

    def evaluate(self, record: "ImageRecord", runtime: "_FilterRuntime") -> bool:
        return runtime.freetext_filter(record, self.text)


@dataclass(frozen=True)
class _AndFilterNode(_FilterNode):
    left: _FilterNode
    right: _FilterNode

    def evaluate(self, record: "ImageRecord", runtime: "_FilterRuntime") -> bool:
        return self.left.evaluate(record, runtime) and self.right.evaluate(record, runtime)


@dataclass(frozen=True)
class _OrFilterNode(_FilterNode):
    left: _FilterNode
    right: _FilterNode

    def evaluate(self, record: "ImageRecord", runtime: "_FilterRuntime") -> bool:
        return self.left.evaluate(record, runtime) or self.right.evaluate(record, runtime)


@dataclass(frozen=True)
class _NotFilterNode(_FilterNode):
    operand: _FilterNode

    def evaluate(self, record: "ImageRecord", runtime: "_FilterRuntime") -> bool:
        return not self.operand.evaluate(record, runtime)


@dataclass(frozen=True)
class _ComparisonFilterNode(_FilterNode):
    operator: str  # "<", ">", "<=", ">="
    value: float

    def evaluate(self, record: "ImageRecord", runtime: "_FilterRuntime") -> bool:
        resolution_mpx = runtime.get_resolution_mpx(record)
        if resolution_mpx is None:
            return False
        if self.operator == "<":
            return resolution_mpx < self.value
        elif self.operator == "<=":
            return resolution_mpx <= self.value
        elif self.operator == ">":
            return resolution_mpx > self.value
        elif self.operator == ">=":
            return resolution_mpx >= self.value
        return False


@dataclass
class _FilterRuntime:
    named_filters: dict[str, Callable[["ImageRecord"], bool]]
    tag_filter: Callable[["ImageRecord", str], bool]
    freetext_filter: Callable[["ImageRecord", str], bool]
    get_resolution_mpx: Callable[["ImageRecord"], float | None]


def _tokenize_filter_expression(expression: str) -> list[_FilterToken]:
    tokens: list[_FilterToken] = []
    index = 0
    length = len(expression)

    while index < length:
        char = expression[index]

        if char.isspace():
            index += 1
            continue

        if char in "&|()": 
            tokens.append(_FilterToken(kind=char, value=char, position=index))
            index += 1
            continue

        if char in "<>":
            start = index
            if char == "<" and index + 1 < length and expression[index + 1] == "=":
                tokens.append(_FilterToken(kind="COMP", value="<=", position=index))
                index += 2
            elif char == ">" and index + 1 < length and expression[index + 1] == "=":
                tokens.append(_FilterToken(kind="COMP", value=">=", position=index))
                index += 2
            else:
                tokens.append(_FilterToken(kind="COMP", value=char, position=index))
                index += 1
            continue

        if char.isdigit() or (char == "." and index + 1 < length and expression[index + 1].isdigit()):
            start = index
            while index < length and (expression[index].isdigit() or expression[index] == "."):
                index += 1
            value = expression[start:index]
            try:
                num_value = float(value)
                tokens.append(_FilterToken(kind="NUMBER", value=value, position=start))
            except ValueError:
                raise FilterSyntaxError(f"Invalid number '{value}' at position {start + 1}.")
            continue

        if char in "!~":
            tokens.append(_FilterToken(kind="NOT", value=char, position=index))
            index += 1
            continue

        if char == '"':
            start = index
            index += 1
            value_chars: list[str] = []
            while index < length:
                current = expression[index]
                if current == "\\":
                    index += 1
                    if index >= length:
                        raise FilterSyntaxError(f"Unfinished escape sequence at position {start + 1}.")
                    value_chars.append(expression[index])
                    index += 1
                    continue
                if current == '"':
                    index += 1
                    break
                value_chars.append(current)
                index += 1
            else:
                raise FilterSyntaxError(f"Missing closing quote for tag at position {start + 1}.")

            tokens.append(_FilterToken(kind="STRING", value="".join(value_chars), position=start))
            continue

        if char == "'":
            start = index
            index += 1
            value_chars: list[str] = []
            while index < length:
                current = expression[index]
                if current == "\\":
                    index += 1
                    if index >= length:
                        raise FilterSyntaxError(f"Unfinished escape sequence at position {start + 1}.")
                    value_chars.append(expression[index])
                    index += 1
                    continue
                if current == "'":
                    index += 1
                    break
                value_chars.append(current)
                index += 1
            else:
                raise FilterSyntaxError(f"Missing closing quote for freetext at position {start + 1}.")

            tokens.append(_FilterToken(kind="FREETEXT", value="".join(value_chars), position=start))
            continue

        start = index
        while index < length and (not expression[index].isspace()) and expression[index] not in "&|()\"!~<>":
            index += 1

        value = expression[start:index]
        if not value:
            raise FilterSyntaxError(f"Unexpected character at position {start + 1}.")
        tokens.append(_FilterToken(kind="NAME", value=value, position=start))

    return tokens


def _parse_filter_expression(expression: str) -> _FilterNode | None:
    tokens = _tokenize_filter_expression(expression)
    if not tokens:
        return None

    position = 0

    def _peek() -> _FilterToken | None:
        if position >= len(tokens):
            return None
        return tokens[position]

    def _consume(expected_kind: str | None = None) -> _FilterToken:
        nonlocal position
        token = _peek()
        if token is None:
            raise FilterSyntaxError("Unexpected end of filter expression.")
        if expected_kind is not None and token.kind != expected_kind:
            raise FilterSyntaxError(
                f"Expected '{expected_kind}' at position {token.position + 1}, got '{token.value}'."
            )
        position += 1
        return token

    def _parse_primary() -> _FilterNode:
        token = _peek()
        if token is None:
            raise FilterSyntaxError("Unexpected end of filter expression.")

        if token.kind == "(":
            _consume("(")
            nested = _parse_or_expression()
            closing = _peek()
            if closing is None or closing.kind != ")":
                at = token.position + 1 if closing is None else closing.position + 1
                raise FilterSyntaxError(f"Missing ')' for group near position {at}.")
            _consume(")")
            return nested

        if token.kind == "NAME":
            name_token = _consume("NAME")
            if name_token.value.casefold() == "resolution":
                comp_token = _peek()
                if comp_token is None or comp_token.kind != "COMP":
                    raise FilterSyntaxError(
                        f"Expected comparison operator after 'resolution' at position {name_token.position + len(name_token.value) + 1}."
                    )
                _consume("COMP")
                num_token = _peek()
                if num_token is None or num_token.kind != "NUMBER":
                    raise FilterSyntaxError(
                        f"Expected number after '{comp_token.value}' at position {comp_token.position + len(comp_token.value) + 1}."
                    )
                _consume("NUMBER")
                try:
                    num_value = float(num_token.value)
                except ValueError:
                    raise FilterSyntaxError(f"Invalid number '{num_token.value}' at position {num_token.position + 1}.")
                return _ComparisonFilterNode(operator=comp_token.value, value=num_value)
            return _NamedFilterNode(name=name_token.value)

        if token.kind == "STRING":
            _consume("STRING")
            return _TagFilterNode(tag=token.value)

        if token.kind == "FREETEXT":
            _consume("FREETEXT")
            return _FreetextFilterNode(text=token.value)

        raise FilterSyntaxError(f"Unexpected token '{token.value}' at position {token.position + 1}.")

    def _parse_not_expression() -> _FilterNode:
        token = _peek()
        if token is not None and token.kind == "NOT":
            _consume("NOT")
            return _NotFilterNode(operand=_parse_not_expression())
        return _parse_primary()

    def _parse_and_expression() -> _FilterNode:
        node = _parse_not_expression()
        while True:
            token = _peek()
            if token is None or token.kind != "&":
                break
            _consume("&")
            node = _AndFilterNode(left=node, right=_parse_not_expression())
        return node

    def _parse_or_expression() -> _FilterNode:
        node = _parse_and_expression()
        while True:
            token = _peek()
            if token is None or token.kind != "|":
                break
            _consume("|")
            node = _OrFilterNode(left=node, right=_parse_and_expression())
        return node

    parsed = _parse_or_expression()
    trailing = _peek()
    if trailing is not None:
        raise FilterSyntaxError(
            f"Unexpected token '{trailing.value}' at position {trailing.position + 1}."
        )
    return parsed
