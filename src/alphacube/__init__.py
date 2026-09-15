"""AlphaCube: one three-axis tensor (S, D, M) = (stock, day, minute) for every
field and every operator, instead of two separate two-axis worlds (ND and DT).

See docs/design.md for the type system, the three faces (DT / ND / NT),
the chunk-phase scheduler, and the precision contract.
"""

__version__ = "0.1.0"
