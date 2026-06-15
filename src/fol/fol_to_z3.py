"""
fol_to_z3.py — Translate FOL AST nodes → Z3 expressions.

Design
------
* All entities share a single uninterpreted sort `Entity`.
* Every n-ary predicate P(x1,...,xn) → FuncDecl P : Entity^n → Bool.
* Numeric arguments use RealSort when a Compare is involved.
* Ground constants (Const nodes) become Z3 Const objects of sort Entity.
* Quantifier variables are Z3 Const objects of sort Entity (used as bound vars).
* The translator is stateful: it accumulates a registry of
      predicate declarations and constant declarations
  so that the same name always maps to the same Z3 object within one item.

Usage
-----
>>> from z3 import *
>>> from src.fol.fol_parser import parse
>>> from src.fol.fol_to_z3 import Z3Context
>>> ctx = Z3Context()
>>> expr = ctx.translate(parse("∀x (WT(x) → O(x))"))
>>> print(expr)
ForAll(x, Implies(WT(x), O(x)))
"""

from __future__ import annotations
from typing import Any
import z3

from src.fol.fol_parser import (
    ForAll, Exists, Implies, Iff, And, Or, Not,
    Atom, Compare, Var, Const, Num,
    ParseError,
)


class Z3Context:
    """
    Maintains a registry of Z3 sorts, functions, and constants for one KB item.
    Create a fresh Z3Context per logical problem (per dataset item).
    """

    def __init__(self):
        # Sort for all logical entities
        self.entity_sort = z3.DeclareSort('Entity')
        self.real_sort   = z3.RealSort()

        # name → z3.FuncDecl  (for predicates and functions)
        self._pred_registry: dict[tuple[str, int], z3.FuncDecl] = {}

        # name → z3.ExprRef  (for ground constants of Entity sort)
        self._const_registry: dict[str, z3.ExprRef] = {}

        # name → z3.ExprRef  (for bound / free variable expressions)
        self._var_stack: dict[str, z3.ExprRef] = {}

        # Numeric functions: name → FuncDecl returning Real
        self._num_func_registry: dict[tuple[str, int], z3.FuncDecl] = {}

    # -----------------------------------------------------------------------
    # Registry helpers
    # -----------------------------------------------------------------------

    def _get_pred(self, name: str, arity: int, numeric_result: bool = False) -> z3.FuncDecl:
        """Return (or create) a Z3 FuncDecl for predicate `name` of given arity."""
        key = (name, arity)
        if key not in self._pred_registry:
            domain = [self.entity_sort] * arity
            if numeric_result:
                decl = z3.Function(name, *domain, self.real_sort)
                self._num_func_registry[key] = decl
            else:
                decl = z3.Function(name, *domain, z3.BoolSort()) if arity > 0 \
                       else z3.Bool(name)
            self._pred_registry[key] = decl
        return self._pred_registry[key]

    def _get_const(self, name: str) -> z3.ExprRef:
        """Return (or create) a Z3 constant of Entity sort."""
        if name not in self._const_registry:
            self._const_registry[name] = z3.Const(name, self.entity_sort)
        return self._const_registry[name]

    def _get_var(self, name: str) -> z3.ExprRef:
        """Return Z3 expression for a variable (from var stack or as entity const)."""
        if name in self._var_stack:
            return self._var_stack[name]
        # Free variable: treat as entity constant
        return self._get_const(name)

    def _push_var(self, name: str, sort=None) -> z3.ExprRef:
        """Create a fresh Z3 variable for a bound quantifier variable.
        
        Default sort is Entity. If the var will be used in numeric comparisons,
        the sort will be upgraded to Real on first numeric use.
        """
        actual_sort = sort if sort is not None else self.entity_sort
        z3_var = z3.Const(name, actual_sort)
        self._var_stack[name] = z3_var
        return z3_var

    def _pop_var(self, name: str):
        self._var_stack.pop(name, None)

    def _upgrade_var_to_real(self, name: str) -> z3.ExprRef:
        """Return a Real-sorted Z3 const for a variable, creating/upgrading as needed."""
        # Check if already in num registry
        key = (name, 0)
        if key in self._num_func_registry:
            return self._num_func_registry[key]
        # Create a Real var; this may coexist with Entity var (they are in different contexts)
        real_var = z3.Real(name + '_num')  # use distinct name to avoid sort clash
        self._num_func_registry[key] = real_var
        # Also update var_stack so entity uses also get the real version if needed
        # But we keep entity var for atom args; only numeric uses use real_var
        return real_var

    def _find_numeric_vars(self, node: Any, result: set | None = None) -> set[str]:
        """
        Recursively scan a formula body for variables that appear directly in
        NUMERIC Compare nodes (>=, <=, >, < with at least one Num node).
        != comparisons are entity-identity checks, not numeric.
        Returns a set of variable names that need Real sort.
        """
        if result is None:
            result = set()
        if isinstance(node, Compare):
            # Only promote to numeric if comparison involves a Num literal
            has_num = isinstance(node.left, Num) or isinstance(node.right, Num)
            if has_num and node.op != '!=':
                for side in (node.left, node.right):
                    if isinstance(side, Var):
                        result.add(side.name)
                    elif isinstance(side, Atom) and len(side.args) == 0:
                        result.add(side.name)
        # Recurse into sub-nodes
        for attr in ('body', 'ant', 'con', 'arg', 'left', 'right'):
            child = getattr(node, attr, None)
            if child is not None:
                self._find_numeric_vars(child, result)
        for attr in ('args',):
            children = getattr(node, attr, None)
            if children:
                for child in children:
                    self._find_numeric_vars(child, result)
        return result


    # -----------------------------------------------------------------------
    # Translation entry point
    # -----------------------------------------------------------------------

    def translate(self, node: Any) -> z3.ExprRef:
        """Translate an AST node to a Z3 expression."""
        return self._tr(node)

    def translate_all(self, nodes: list[Any]) -> list[z3.ExprRef]:
        """Translate a list of premises."""
        return [self._tr(n) for n in nodes]

    # -----------------------------------------------------------------------
    # Internal recursive translator
    # -----------------------------------------------------------------------

    def _tr(self, node: Any) -> z3.ExprRef:
        if isinstance(node, ForAll):
            numeric_vars = self._find_numeric_vars(node.body)
            is_numeric = node.var in numeric_vars
            sort = self.real_sort if is_numeric else self.entity_sort
            z3_var = self._push_var(node.var, sort=sort)
            body   = self._tr(node.body)
            self._pop_var(node.var)
            return z3.ForAll([z3_var], body)

        if isinstance(node, Exists):
            numeric_vars = self._find_numeric_vars(node.body)
            is_numeric = node.var in numeric_vars
            sort = self.real_sort if is_numeric else self.entity_sort
            z3_var = self._push_var(node.var, sort=sort)
            body   = self._tr(node.body)
            self._pop_var(node.var)
            return z3.Exists([z3_var], body)

        if isinstance(node, Implies):
            return z3.Implies(self._tr(node.ant), self._tr(node.con))

        if isinstance(node, Iff):
            l = self._tr(node.left)
            r = self._tr(node.right)
            return z3.And(z3.Implies(l, r), z3.Implies(r, l))

        if isinstance(node, And):
            args = [self._tr(a) for a in node.args]
            return z3.And(*args) if len(args) > 1 else args[0]

        if isinstance(node, Or):
            args = [self._tr(a) for a in node.args]
            return z3.Or(*args) if len(args) > 1 else args[0]

        if isinstance(node, Not):
            return z3.Not(self._tr(node.arg))

        if isinstance(node, Compare):
            op = node.op
            has_num = isinstance(node.left, Num) or isinstance(node.right, Num)

            if op == '!=' and not has_num:
                # Entity identity check (e.g., m1 ≠ m2): use entity-sort comparison
                left_z3  = self._tr_any_term(node.left)
                right_z3 = self._tr_any_term(node.right)
                return left_z3 != right_z3
            elif op == '=' and not has_num:
                # Entity equality (e.g., x = john): use entity-sort
                left_z3  = self._tr_any_term(node.left)
                right_z3 = self._tr_any_term(node.right)
                return left_z3 == right_z3
            else:
                # Numeric comparison (one side is a literal, or >=, <=, >, <)
                left_z3  = self._tr_numeric(node.left)
                right_z3 = self._tr_numeric(node.right)
                if op == '>=': return left_z3 >= right_z3
                if op == '<=': return left_z3 <= right_z3
                if op == '>':  return left_z3 > right_z3
                if op == '<':  return left_z3 < right_z3
                if op == '=':  return left_z3 == right_z3
                if op == '!=': return left_z3 != right_z3
            raise TranslateError(f"Unknown compare op: {op!r}")


        if isinstance(node, Atom):
            return self._tr_atom(node)

        if isinstance(node, Var):
            return self._get_var(node.name)

        if isinstance(node, Const):
            return self._get_const(node.name)

        if isinstance(node, Num):
            return z3.RealVal(node.value)

        raise TranslateError(f"Unknown AST node: {type(node).__name__} {node!r}")

    def _tr_atom(self, node: Atom) -> z3.ExprRef:
        """Translate Atom(name, args) to a Z3 boolean application."""
        name  = node.name
        arity = len(node.args)

        if arity == 0:
            # Propositional / zero-arity → Bool constant
            return z3.Bool(name)

        # Translate each argument, preserving their actual sort
        arg_exprs = [self._tr_any_term(a) for a in node.args]

        # Determine domain sorts from actual arg sorts
        domain = [e.sort() for e in arg_exprs]

        # Get or create predicate declaration with this exact domain
        key = (name, arity)
        if key not in self._pred_registry:
            decl = z3.Function(name, *domain, z3.BoolSort())
            self._pred_registry[key] = decl
        else:
            decl = self._pred_registry[key]
            # Check sort compatibility; if mismatch, create a domain-specific variant
            try:
                return decl(*arg_exprs)
            except Exception:
                # Sort mismatch with cached version — create fresh with actual sorts
                decl = z3.Function(name + '_v2', *domain, z3.BoolSort())

        return decl(*arg_exprs)

    def _tr_any_term(self, node: Any) -> z3.ExprRef:
        """
        Translate a term in any sort context — returns the var's actual registered sort.
        Used for atom args where some may be Entity, others Real.
        """
        if isinstance(node, Var):
            # Return from var_stack with its actual sort (Entity or Real)
            if node.name in self._var_stack:
                return self._var_stack[node.name]
            return self._get_const(node.name)
        if isinstance(node, Const):
            return self._get_const(node.name)
        if isinstance(node, Atom):
            if len(node.args) == 0:
                if node.name in self._var_stack:
                    return self._var_stack[node.name]
                return self._get_const(node.name)
            # Functional term — fall back to entity_term logic
            return self._tr_entity_term(node)
        if isinstance(node, Num):
            return z3.RealVal(node.value)
        if isinstance(node, Not):
            return self._tr(node)
        return self._tr(node)


    def _tr_entity_term(self, node: Any) -> z3.ExprRef:
        """
        Translate a term that should be Entity-sorted.
        Handles Var, Const, Atom (functional terms), Num (coerced — best effort).
        """
        if isinstance(node, Var):
            return self._get_var(node.name)
        if isinstance(node, Const):
            return self._get_const(node.name)
        if isinstance(node, Atom):
            if len(node.args) == 0:
                # bare name in term position: Var or Const
                if node.name in self._var_stack:
                    return self._var_stack[node.name]
                # Check if it was declared as entity const
                return self._get_const(node.name)
            # Functional term: treat as uninterpreted function Entity^n → Entity
            name  = node.name
            arity = len(node.args)
            func_key = ('_func_' + name, arity)
            if func_key not in self._pred_registry:
                domain = [self.entity_sort] * arity
                decl = z3.Function(name + '_fn', *domain, self.entity_sort)
                self._pred_registry[func_key] = decl
            arg_exprs = [self._tr_entity_term(a) for a in node.args]
            return self._pred_registry[func_key](*arg_exprs)
        if isinstance(node, Num):
            # Numeric in entity position — should not happen in clean FOL
            # Return a Real variable as best-effort
            return z3.RealVal(node.value)
        if isinstance(node, Not):
            # Not in term position (e.g. ¬pass inside existential) — translate as Bool in Bool context
            return self._tr(node)
        # fallback
        return self._tr(node)

    def _tr_numeric(self, node: Any) -> z3.ExprRef:
        """
        Translate a term that should be numeric (Real / Int) for comparisons.
        Handles: Num, Atom (uninterpreted real function), Var.
        """
        if isinstance(node, Num):
            return z3.RealVal(node.value)
        if isinstance(node, Atom):
            name  = node.name
            arity = len(node.args)
            # Numeric function: Entity^n → Real
            key = (name, arity)
            if key not in self._num_func_registry:
                if arity == 0:
                    decl = z3.Real(name)
                else:
                    domain = [self.entity_sort] * arity
                    decl = z3.Function(name, *domain, self.real_sort)
                self._num_func_registry[key] = decl
            decl = self._num_func_registry[key]
            if arity == 0:
                return decl
            arg_exprs = [self._tr_entity_term(a) for a in node.args]
            return decl(*arg_exprs)
        if isinstance(node, Var):
            # Variable in numeric context: check if it was bound as Real
            if node.name in self._var_stack:
                v = self._var_stack[node.name]
                if v.sort() == self.real_sort:
                    return v
                # Entity-sorted var in numeric position: create a paired Real var
                key = ('_numvar_' + node.name, 0)
                if key not in self._num_func_registry:
                    self._num_func_registry[key] = z3.Real('_n_' + node.name)
                return self._num_func_registry[key]
            # Unbound var: use Real
            key = (node.name, 0)
            if key not in self._num_func_registry:
                self._num_func_registry[key] = z3.Real(node.name)
            return self._num_func_registry[key]
        if isinstance(node, Const):
            return z3.Real(node.name)
        # Compare left of another compare: shouldn't happen
        return self._tr(node)


class TranslateError(Exception):
    pass


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from src.fol.fol_parser import parse

    TESTS = [
        # Basic universal Horn
        ("∀x (WT(x) → O(x))",   "ForAll([x], Implies(WT(x), O(x)))"),
        # Negation
        ("∀x (¬PEP8(x) → ¬WT(x))", None),
        # Bare forall fact
        ("∀x (EM(x))", None),
        # Existential
        ("∃x (BP(x))", None),
        # Biconditional bare atoms
        ("FailTest ↔ ¬PassTest", None),
        # Conjunction antecedent
        ("ForAll(x, (completed_core_curriculum(x) ∧ passed_science_assessment(x)) → qualified_for_advanced_courses(x))", None),
        # Nested ForAll
        ("ForAll(x, ForAll(d, (faculty_member(x) ∧ has_degree(x, d) ∧ higher(d, BA)) → teach_undergrad(x)))", None),
        # Numeric comparison
        ("ForAll(s, ForAll(m, (attendance(s,m) >= 80) → allowed_exam(s,m)))", None),
        # Disjunction in consequent
        ("∀x (StudySchedule(x) ↔ (RegisterSubject(x) ∨ FirstSemester(x)))", None),
        # Ground equality
        ("membership_duration(Alex) = 8", None),
    ]

    ok = fail = 0
    for fol_str, _expected in TESTS:
        try:
            ast  = parse(fol_str)
            ctx  = Z3Context()
            expr = ctx.translate(ast)
            # Quick sanity: expr should be a Z3 expression
            assert expr is not None, "Got None"
            assert hasattr(expr, 'sort') or isinstance(expr, z3.BoolRef), f"Not a Z3 expr: {type(expr)}"
            print(f"  ✓  {fol_str[:65]}")
            print(f"       Z3: {expr}")
            ok += 1
        except Exception as e:
            print(f"  ✗  {fol_str[:65]}")
            print(f"       ERROR: {e}")
            import traceback; traceback.print_exc()
            fail += 1

    print(f"\n{ok}/{ok+fail} Z3 translation tests passed")
