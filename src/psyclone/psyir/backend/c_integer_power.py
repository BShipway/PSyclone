# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2026, Science and Technology Facilities Council.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------

"""Write a Fortran integer power as the multiplications gfortran makes.

Fortran's ``**`` with a constant integer exponent is not the C library's
``pow``. gfortran expands the exponent into a chain of multiplications, each
of which IEEE-754 rounds correctly; ``pow`` is a library function that is not
required to be correctly rounded, and glibc's is not. The two disagree in the
last bit for some operands, and a region that called ``pow`` where the kernel
wrote ``**`` drifts from the Fortran it replaced.

That is not a tolerance question. The C16_MG whole-model checksums have zero
tolerance, and ``ffsl_fourth_dldz_code``, whose inlined helper computes
``edge_height ** 2`` and ``edge_height ** 3``, moved them by about one part
in ``1e12``. Rewriting those two calls by hand as ``(x * x)``
and ``((x * x) * x)`` restored the reference exactly, with nothing else
changed; that experiment is what this module generalises.

The chain written here is the binary method -- square the running power and
multiply it into the result wherever the exponent has a bit set -- which is
what gfortran's own front end emits (``gfc_conv_cst_int_power``). It was
measured rather than read off: 200000 pseudo-random operands, exponents two to
eight, compared bit for bit against gfortran 14.2.0's answer for the same
``**``.

Three things that measurement found bound what this module can promise, and
none of them is obvious.

* ``pow`` is inexact at exponent **two**, not only at three. It differs from
  ``x * x`` for about one operand in a thousand and from ``((x * x) * x)`` for
  about a quarter of them, which is why the fourth-order kernel drifted where
  its third-order sibling, whose only power is ``** 2``, did not. The sibling
  agreed by luck rather than by ``pow(x, 2.0)`` being exact.

* gfortran's answer for exponents **five and six** depends on the optimisation
  level, so no single rendering reproduces it everywhere. At ``-O0`` and
  ``-Og`` the front end's binary method reaches the object code; from ``-O1``
  the middle end re-expands ``__builtin_powi`` through GCC's ``powi_table``
  addition chains (``tree-ssa-math-opts.cc``), which associate those two
  exponents differently, and ``-Ofast`` gives one of each. Every other
  exponent up to eight is the same tree at every level, because the two
  methods coincide there. The binary method is the one written because
  LFRic's ``fast-debug`` profile, which the checksum gate builds at, is
  ``-Og``.

* Above eight nothing has been measured, so above eight the writer keeps
  ``pow`` and this module says nothing about it.

"""

from psyclone.psyir.nodes import Literal, UnaryOperation
from psyclone.psyir.symbols import ScalarType


#: The largest exponent whose tree has been compared with gfortran's, and so
#: the largest one written as a tree. Nothing above it is known to be right.
POWER_TREE_LIMIT = 8


def literal_exponent(node):
    """Read the exponent of a ``**`` as a Python integer, if it is one.

    The Fortran front end writes ``x ** (-2)`` as a unary minus over the
    literal two rather than as a signed literal, so a sign is unwrapped here
    rather than left to :py:meth:`Literal.value`.

    :param node: the right-hand operand of the power.
    :type node: :py:class:`psyclone.psyir.nodes.DataNode`

    :returns: its value, or ``None`` if it is not an integer literal.
    :rtype: Optional[int]

    """
    sign = 1
    if isinstance(node, UnaryOperation):
        if node.operator == UnaryOperation.Operator.MINUS:
            sign, node = -1, node.children[0]
        elif node.operator == UnaryOperation.Operator.PLUS:
            node = node.children[0]
    if not (isinstance(node, Literal)
            and node.datatype.intrinsic == ScalarType.Intrinsic.INTEGER):
        return None
    return sign * int(node.value)


def power_tree(base, exponent):
    """Write ``base`` raised to a positive ``exponent`` as products.

    The association is the assertion, not the number of multiplications: a
    tree over the same operands in a different order rounds differently, and
    only one order is gfortran's.

    :param str base: the C text of the operand being raised.
    :param int exponent: the power to raise it to, one or more.

    :returns: the C text of the product tree.
    :rtype: str

    """
    result = None
    term = base
    while exponent:
        if exponent & 1:
            result = term if result is None else f"({term} * {result})"
        exponent >>= 1
        if exponent:
            term = f"({term} * {term})"
    return result


def integer_power(base, exponent, base_is_real):
    """Write ``base ** exponent`` as products, if that is what Fortran does.

    A negative exponent is written over a real base only. Fortran evaluates an
    integer raised to a negative power as an integer, which is zero for every
    base but one and minus one, so a reciprocal would be a different answer
    rather than a better-rounded one; and an exponent whose base has no
    datatype the writer can read is not known to be real, so it keeps ``pow``
    as it did before.

    The reciprocal's numerator is the integer ``1`` rather than ``1.0`` so
    that C++'s arithmetic conversions give it the base's own type. A ``1.0``
    would compute a single-precision reciprocal in double and round twice.

    :param str base: the C text of the operand being raised.
    :param exponent: the right-hand operand of the power.
    :type exponent: :py:class:`psyclone.psyir.nodes.DataNode`
    :param bool base_is_real: whether the base is known to be a real scalar.

    :returns: the C text of the product tree, or ``None`` if the power is one
        the writer should leave to ``pow``.
    :rtype: Optional[str]

    """
    value = literal_exponent(exponent)
    if value is None or not 1 <= abs(value) <= POWER_TREE_LIMIT:
        return None
    if value > 0:
        return power_tree(base, value)
    if not base_is_real:
        return None
    return f"(1 / {power_tree(base, -value)})"
