"""
fol_parser.py — First-Order Logic parser for the Oracle Harness.

Accepts TWO notation styles found in the dataset:
  Unicode :  ∀x (P(x) → Q(x)),  ∃x (P(x)),  ¬P(x),  P ∧ Q,  P ∨ Q,  P ↔ Q
  Functional: ForAll(x, P(x) → Q(x)), Exists(x, P(x)), nested ForAll

Both are normalised to the same internal AST.

Usage
-----
>>> from src.fol.fol_parser import parse, ForAll, Exists, Implies, Not, Atom, Var
>>> f = parse("∀x (WT(x) → O(x))")
>>> f
ForAll(var='x', body=Implies(ant=Atom(name='WT', args=(Var(name='x'),)), con=Atom(name='O', args=(Var(name='x'),))))

>>> parse("ForAll(x, P(x) → Q(x))")
ForAll(var='x', body=Implies(ant=Atom(name='P', args=(Var(name='x'),)), con=Atom(name='Q', args=(Var(name='x'),))))

>>> parse("∃x (BP(x))")
Exists(var='x', body=Atom(name='BP', args=(Var(name='x'),)))

>>> parse("¬WT(x)")
Not(arg=Atom(name='WT', args=(Var(name='x'),)))

>>> parse("FailTest ↔ ¬PassTest")
Iff(left=Atom(name='FailTest', args=()), right=Not(arg=Atom(name='PassTest', args=())))
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import re


# ---------------------------------------------------------------------------
# AST node definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForAll:
    var: str
    body: Any

@dataclass(frozen=True)
class Exists:
    var: str
    body: Any

@dataclass(frozen=True)
class Implies:
    ant: Any
    con: Any

@dataclass(frozen=True)
class Iff:
    left: Any
    right: Any

@dataclass(frozen=True)
class And:
    args: tuple

@dataclass(frozen=True)
class Or:
    args: tuple

@dataclass(frozen=True)
class Not:
    arg: Any

@dataclass(frozen=True)
class Atom:
    """A predicate application: name(arg1, arg2, ...)  or bare name (arity 0)."""
    name: str
    args: tuple

@dataclass(frozen=True)
class Compare:
    """Numeric comparison: left op right  where op ∈ {>=, <=, >, <, =, !=}."""
    op: str
    left: Any
    right: Any

@dataclass(frozen=True)
class Var:
    name: str

@dataclass(frozen=True)
class Const:
    """Ground constant (starts with uppercase or is a quoted string)."""
    name: str

@dataclass(frozen=True)
class Num:
    value: float


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<SPACES>[ \t\n]+)                          # whitespace (skip)
    |(?P<FORALL_KW>ForAll)                        # keyword ForAll
    |(?P<EXISTS_KW>Exists)                        # keyword Exists
    |(?P<NOT_KW>Not)                              # keyword Not  (rare)
    |(?P<UNICODE_FORALL>[\u2200])                      # \u2200
    |(?P<UNICODE_EXISTS>[\u2203])                      # \u2203
    |(?P<UNICODE_NEG>[\u00ac])                         # \u00ac
    |(?P<UNICODE_AND>[\u2227])                         # \u2227
    |(?P<UNICODE_OR>[\u2228])                          # \u2228
    |(?P<UNICODE_IFF>[\u2194])                         # \u2194
    |(?P<UNICODE_IMPL>[\u2192])                        # \u2192
    |(?P<UNICODE_NEQ>[\u2260])                         # \u2260 \u2260
    |(?P<UNICODE_GEQ>[\u2265])                         # \u2265 \u2265
    |(?P<UNICODE_LEQ>[\u2264])                         # \u2264 \u2264
    |(?P<BICOND><->)                              # <->
    |(?P<IMPL>->)                                 # ->
    |(?P<GEQ>>=)                                  # >=
    |(?P<LEQ><=)                                  # <=
    |(?P<NEQ>!=)                                  # !=
    |(?P<GT>>)                                    # >
    |(?P<LT><)                                    # <
    |(?P<EQ>=)                                    # =  (ground equality / compare)
    |(?P<BANG>!)                                  # ! (ascii NOT)
    |(?P<COMMA>,)                                 # ,
    |(?P<LPAREN>\()                               # (
    |(?P<RPAREN>\))                               # )
    |(?P<AND_KW>and\b)                            # and (ascii)
    |(?P<OR_KW>or\b)                              # or  (ascii)
    |(?P<NUM>-?\d+(?:\.\d+)?)                     # number literal
    |(?P<NAME>[A-Za-z_][A-Za-z0-9_']*)           # identifier
    """,
    re.VERBOSE | re.UNICODE,
)

def tokenize(text: str) -> list[tuple[str, str]]:
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind == 'SPACES':
            continue
        tokens.append((kind, m.group()))
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    # --- helpers -----------------------------------------------------------

    def peek(self) -> tuple[str, str] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def peek_kind(self) -> str | None:
        t = self.peek()
        return t[0] if t else None

    def consume(self, kind: str | None = None) -> tuple[str, str]:
        if self.pos >= len(self.tokens): raise ParseError("Unexpected end of input")
        t = self.tokens[self.pos]
        if kind and t[0] != kind:
            raise ParseError(
                f"Expected {kind!r} but got {t[0]!r} ({t[1]!r}) at pos {self.pos}"
            )
        self.pos += 1
        return t

    def match(self, *kinds: str) -> bool:
        return self.peek_kind() in kinds

    # --- grammar -----------------------------------------------------------
    # Precedence (low → high):
    #   iff (↔ / <->)  →  implies (→ / ->)  →  or (∨)  →  and (∧)  →  not (¬/!)  →  atom/quantifier

    def parse_formula(self) -> Any:
        return self.parse_iff()

    def parse_iff(self) -> Any:
        left = self.parse_implies()
        while self.match('UNICODE_IFF', 'BICOND'):
            self.consume()
            right = self.parse_implies()
            left = Iff(left, right)
        return left

    def parse_implies(self) -> Any:
        left = self.parse_or()
        while self.match('UNICODE_IMPL', 'IMPL'):
            self.consume()
            right = self.parse_or()
            left = Implies(left, right)
        return left

    def parse_or(self) -> Any:
        left = self.parse_and()
        args = [left]
        while self.match('UNICODE_OR', 'OR_KW'):
            self.consume()
            args.append(self.parse_and())
        if len(args) == 1:
            return args[0]
        return Or(tuple(args))

    def parse_and(self) -> Any:
        left = self.parse_unary()
        args = [left]
        while self.match('UNICODE_AND', 'AND_KW', 'COMMA'):
            # Comma inside a ForAll body can be ∧ — but inside term args it's a separator.
            # We only interpret COMMA as AND at the formula level (not inside Atom args).
            if self.peek_kind() == 'COMMA':
                # peek ahead: if next-next is a quantifier / formula opener → it's AND
                # if next-next looks like a NAME followed by RPAREN → term separator
                # Heuristic: treat comma as AND only if the next token after comma
                # is a quantifier keyword or ¬ or NAME followed by LPAREN or plain bool.
                # For safety, we do NOT consume COMMA here — let it fall through.
                break
            self.consume()
            args.append(self.parse_unary())
        if len(args) == 1:
            return args[0]
        return And(tuple(args))

    def parse_unary(self) -> Any:
        if self.match('UNICODE_NEG', 'BANG'):
            self.consume()
            arg = self.parse_unary()
            return Not(arg)
        if self.match('NOT_KW'):
            self.consume()
            self.consume('LPAREN')
            arg = self.parse_formula()
            self.consume('RPAREN')
            return Not(arg)
        return self.parse_primary()

    def parse_primary(self) -> Any:
        k = self.peek_kind()
        # Quantifiers (unicode)
        if k == 'UNICODE_FORALL':
            return self.parse_forall_unicode()
        if k == 'UNICODE_EXISTS':
            return self.parse_exists_unicode()
        # Quantifiers (functional)
        if k == 'FORALL_KW':
            return self.parse_forall_functional()
        if k == 'EXISTS_KW':
            return self.parse_exists_functional()
        # Parenthesised sub-formula
        if k == 'LPAREN':
            self.consume('LPAREN')
            inner = self.parse_formula()
            self.consume('RPAREN')
            return inner
        # Number
        if k == 'NUM':
            _, val = self.consume('NUM')
            return Num(float(val))
        # NAME: either predicate/constant or start of compare
        if k == 'NAME':
            return self.parse_name_or_compare()
        raise ParseError(
            f"Unexpected token {self.peek()!r} at pos {self.pos} "
            f"(remaining: {self.tokens[self.pos:self.pos+5]})"
        )

    # --- quantifier parsers ------------------------------------------------

    def parse_forall_unicode(self) -> Any:
        """∀x ∀y (body)  or  ∀x(body)"""
        self.consume('UNICODE_FORALL')
        var = self._consume_var()
        # optional comma between multiple ∀ vars
        if self.match('UNICODE_FORALL'):
            return ForAll(var, self.parse_forall_unicode())
        if self.match('UNICODE_EXISTS'):
            return ForAll(var, self.parse_exists_unicode())
        # consume optional whitespace wrapper paren
        body = self._parse_optional_paren_formula()
        return ForAll(var, body)

    def parse_exists_unicode(self) -> Any:
        """∃x (body)  or  ∃x1, ∃x2, ∃x3 (body)"""
        self.consume('UNICODE_EXISTS')
        var = self._consume_var()
        # chain: ∃x1, ∃x2, ... body  (comma-separated exists)
        if self.match('COMMA'):
            nxt_pos = self.pos + 1
            if nxt_pos < len(self.tokens) and self.tokens[nxt_pos][0] in ('UNICODE_EXISTS', 'EXISTS_KW'):
                self.consume('COMMA')
                return Exists(var, self.parse_exists_unicode())
        # No more chained exists → parse body (may be the rest of the And after comma)
        if self.match('COMMA'):
            # ∃m3, rest_of_formula → treat rest as body (conjunction context)
            self.consume('COMMA')
            body = self.parse_formula()
            return Exists(var, body)
        body = self._parse_optional_paren_formula()
        return Exists(var, body)

    def parse_forall_functional(self) -> Any:
        """ForAll(x, body)  or  ForAll(x, ForAll(y, body))"""
        self.consume('FORALL_KW')
        self.consume('LPAREN')
        var = self._consume_var()
        self.consume('COMMA')
        body = self.parse_formula()
        self.consume('RPAREN')
        return ForAll(var, body)

    def parse_exists_functional(self) -> Any:
        """Exists(x, body)  or  Exists(x1, Exists(x2, body))"""
        self.consume('EXISTS_KW')
        self.consume('LPAREN')
        var = self._consume_var()
        self.consume('COMMA')
        body = self.parse_formula()
        self.consume('RPAREN')
        return Exists(var, body)

    # --- atom / compare parser --------------------------------------------

    def parse_name_or_compare(self) -> Any:
        """
        Handles:
          pred(args...)           → Atom
          bare_name               → Atom (arity-0)
          func(args) >= 8.5       → Compare
          NAME = 8                → Compare (ground equality)
        """
        _, name = self.consume('NAME')
        # predicate application with args?
        if self.match('LPAREN'):
            self.consume('LPAREN')
            args = self._parse_term_list()
            self.consume('RPAREN')
            atom = Atom(name, tuple(args))
        else:
            # bare name: could be constant or variable
            # Convention: starts with lowercase → Var, uppercase → Const
            # But in FOL strings many predicates are bare arity-0 atoms (e.g. "FailTest")
            # We treat bare names as Atom(name, ()) — the Z3 layer will decide sort
            atom = Atom(name, ())

        # possible comparison after atom
        if self.match('GEQ', 'LEQ', 'GT', 'LT', 'EQ', 'NEQ', 'UNICODE_NEQ', 'UNICODE_GEQ', 'UNICODE_LEQ', 'UNICODE_IFF'):
            # Be careful: UNICODE_IFF is ↔ not ≥. But EQ is =.
            if self.peek_kind() == 'UNICODE_IFF':
                # Actually this is iff not compare, return atom and let upper parser handle ↔
                return atom
            op_kind, op_str = self.consume()
            # Normalize unicode comparison operators
            if op_kind == 'UNICODE_NEQ': op_str = '!='
            elif op_kind == 'UNICODE_GEQ': op_str = '>='
            elif op_kind == 'UNICODE_LEQ': op_str = '<='
            rhs = self._parse_term()
            return Compare(op_str, atom, rhs)

        return atom

    def _parse_term(self) -> Any:
        """Parse a term: NAME(args) | bare NAME | NUM."""
        k = self.peek_kind()
        if k == 'NUM':
            _, val = self.consume('NUM')
            return Num(float(val))
        if k == 'UNICODE_NEG' or k == 'BANG':
            # negated term inside compare (e.g. ¬pass)
            self.consume()
            inner = self._parse_term()
            return Not(inner)
        if k == 'NAME':
            _, name = self.consume('NAME')
            if self.match('LPAREN'):
                self.consume('LPAREN')
                args = self._parse_term_list()
                self.consume('RPAREN')
                return Atom(name, tuple(args))
            return Atom(name, ())
        raise ParseError(f"Expected term but got {self.peek()!r}")

    def _parse_term_list(self) -> list:
        """Parse comma-separated terms inside Atom args."""
        if self.match('RPAREN'):
            return []
        terms = [self._parse_term()]
        while self.match('COMMA'):
            # Peek: if next is a quantifier or formula-level thing, stop
            nxt = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else None
            if nxt and nxt[0] in ('UNICODE_FORALL', 'UNICODE_EXISTS', 'FORALL_KW', 'EXISTS_KW',
                                   'UNICODE_AND', 'UNICODE_OR', 'UNICODE_IFF', 'UNICODE_IMPL',
                                   'UNICODE_NEG', 'BANG', 'NOT_KW'):
                break
            self.consume('COMMA')
            if self.match('RPAREN'):
                break
            terms.append(self._parse_term())
        return terms

    def _consume_var(self) -> str:
        """Consume a variable name token."""
        if self.peek_kind() == 'NAME':
            _, name = self.consume('NAME')
            return name
        raise ParseError(f"Expected variable name, got {self.peek()!r}")

    def _parse_optional_paren_formula(self) -> Any:
        """Parse (formula) or formula — used after quantifier var."""
        if self.match('LPAREN'):
            self.consume('LPAREN')
            inner = self.parse_formula()
            self.consume('RPAREN')
            return inner
        return self.parse_formula()


class ParseError(Exception):
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(text: str) -> Any:
    """
    Parse a FOL string into an AST.

    >>> parse("∀x (WT(x) → O(x))")
    ForAll(var='x', body=Implies(ant=Atom(name='WT', args=(Var(name='x'),)), con=Atom(name='O', args=(Var(name='x'),))))

    Hmm, Var vs Const: after parsing, bare lowercase inside Atom args are Var,
    uppercase are Const.  We do this by post-processing.
    """
    tokens = tokenize(text)
    p = _Parser(tokens)
    result = p.parse_formula()
    if p.pos < len(p.tokens):
        # remaining tokens — try to continue (some formulas have trailing junk)
        pass
    return _resolve_vars_consts(result, bound_vars=frozenset())


def _resolve_vars_consts(node: Any, bound_vars: frozenset) -> Any:
    """
    Walk the AST and replace Atom(name, ()) with Var(name) or Const(name)
    when they appear as *terms* (inside Atom.args or Compare operands).

    Atoms at the formula level remain Atom(name, ()).
    """
    if isinstance(node, ForAll):
        new_bound = bound_vars | {node.var}
        return ForAll(node.var, _resolve_vars_consts(node.body, new_bound))
    if isinstance(node, Exists):
        new_bound = bound_vars | {node.var}
        return Exists(node.var, _resolve_vars_consts(node.body, new_bound))
    if isinstance(node, Implies):
        return Implies(
            _resolve_vars_consts(node.ant, bound_vars),
            _resolve_vars_consts(node.con, bound_vars),
        )
    if isinstance(node, Iff):
        return Iff(
            _resolve_vars_consts(node.left, bound_vars),
            _resolve_vars_consts(node.right, bound_vars),
        )
    if isinstance(node, And):
        return And(tuple(_resolve_vars_consts(a, bound_vars) for a in node.args))
    if isinstance(node, Or):
        return Or(tuple(_resolve_vars_consts(a, bound_vars) for a in node.args))
    if isinstance(node, Not):
        return Not(_resolve_vars_consts(node.arg, bound_vars))
    if isinstance(node, Atom):
        new_args = tuple(_resolve_term(a, bound_vars) for a in node.args)
        return Atom(node.name, new_args)
    if isinstance(node, Compare):
        return Compare(
            node.op,
            _resolve_term(node.left, bound_vars),
            _resolve_term(node.right, bound_vars),
        )
    return node


def _resolve_term(node: Any, bound_vars: frozenset) -> Any:
    """Resolve a term node: bare Atom → Var if bound, else Const."""
    if isinstance(node, Atom) and len(node.args) == 0:
        name = node.name
        # If it's a bound variable name → Var
        if name in bound_vars:
            return Var(name)
        # If all lowercase (or single lowercase letter) → could be free var;
        # but we conservatively treat non-bound as Const for ground atoms.
        # The reasoner will handle grounding.
        if name[0].islower() and name not in bound_vars:
            return Var(name)  # unbound lowercase = implicitly universally quantified
        return Const(name)
    if isinstance(node, Atom):
        new_args = tuple(_resolve_term(a, bound_vars) for a in node.args)
        return Atom(node.name, new_args)
    if isinstance(node, Num):
        return node
    return node


# ---------------------------------------------------------------------------
# Collect predicates & constants from an AST
# ---------------------------------------------------------------------------

def collect_predicates(node: Any, result: dict | None = None) -> dict[str, int]:
    """Return {predicate_name: arity} from an AST."""
    if result is None:
        result = {}
    if isinstance(node, (ForAll, Exists)):
        collect_predicates(node.body, result)
    elif isinstance(node, (Implies, Iff)):
        for child in (node.ant, node.con) if isinstance(node, Implies) else (node.left, node.right):
            collect_predicates(child, result)
    elif isinstance(node, (And, Or)):
        for a in node.args:
            collect_predicates(a, result)
    elif isinstance(node, Not):
        collect_predicates(node.arg, result)
    elif isinstance(node, Atom):
        result[node.name] = len(node.args)
        for a in node.args:
            collect_predicates(a, result)
    elif isinstance(node, Compare):
        collect_predicates(node.left, result)
        collect_predicates(node.right, result)
    return result


def collect_constants(node: Any, result: set | None = None) -> set[str]:
    """Return all ground Const names in an AST."""
    if result is None:
        result = set()
    if isinstance(node, Const):
        result.add(node.name)
    elif isinstance(node, (ForAll, Exists)):
        collect_constants(node.body, result)
    elif isinstance(node, (Implies, Iff)):
        for child in (node.ant, node.con) if isinstance(node, Implies) else (node.left, node.right):
            collect_constants(child, result)
    elif isinstance(node, (And, Or)):
        for a in node.args:
            collect_constants(a, result)
    elif isinstance(node, Not):
        collect_constants(node.arg, result)
    elif isinstance(node, Atom):
        for a in node.args:
            collect_constants(a, result)
    elif isinstance(node, Compare):
        collect_constants(node.left, result)
        collect_constants(node.right, result)
    return result


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    TESTS = [
        # unicode forall + implies
        "∀x (WT(x) → O(x))",
        # unicode negation in antecedent + consequent
        "∀x (¬PEP8(x) → ¬WT(x))",
        # bare universal fact
        "∀x (EM(x))",
        # existential
        "∃x (BP(x))",
        # biconditional bare atoms
        "FailTest ↔ ¬PassTest",
        # functional ForAll + conjunction
        "ForAll(x, (completed_core_curriculum(x) ∧ passed_science_assessment(x)) → qualified_for_advanced_courses(x))",
        # nested ForAll functional
        "ForAll(x, ForAll(d, (faculty_member(x) ∧ has_degree(x, d) ∧ higher(d, BA)) → teach_undergrad(x)))",
        # numeric compare
        "ForAll(s, ForAll(m, (attendance(s,m) >= 80) → allowed_exam(s,m)))",
        # disjunction in consequent
        "ForAll(s, analyze(s,'GovernmentPolicies') → (debate(s) ∨ write_essay(s)))",
        # biconditional with ∨
        "∀x (StudySchedule(x) ↔ (RegisterSubject(x) ∨ FirstSemester(x)))",
        # ground equality
        "membership_duration(Alex) = 8",
        # existential chained
        "ForAll(s, (∃m1, ∃m2, ∃m3, m1 ≠ m2 ∧ m2 ≠ m3 ∧ m1 ≠ m3 ∧ grade(s,m1) > 8.5 ∧ grade(s,m2) > 8.5 ∧ grade(s,m3) > 8.5) → scholarship(s))",
        # ascii arrow
        "∀x (WS(x) → O(x))",
        # ascii ForAll with comparison
        "ForAll(s, ForAll(m, (attendance(s,m) < 50) → ¬allowed_exam(s,m)))",
    ]

    ok = 0
    fail = 0
    for fol_str in TESTS:
        try:
            ast = parse(fol_str)
            print(f"  ✓  {fol_str[:70]}")
            ok += 1
        except Exception as e:
            print(f"  ✗  {fol_str[:70]}")
            print(f"       ERROR: {e}")
            fail += 1

    print(f"\n{ok}/{ok+fail} tests passed")
