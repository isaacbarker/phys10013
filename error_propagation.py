import sympy as sp
import math

def propagate_error_pd(f: sp.Expr, vars: dict[sp.Symbol, tuple[float, float]]) -> tuple[float, float]:
    '''
    Error Propagation (Partial Differentiation)

    Takes an expr and propagates symbols with value and errors.
    Returns a tuple of (value, error)
    '''

    symbols = sorted(vars.keys(), key=lambda s: s.name) # ensures all symbols are ordered correctly
    values = [vars[s][0] for s in symbols]
    errors = [vars[s][1] for s in symbols]

    uncertainty_squared = []
    subs = dict(zip(symbols, values))

    # use partial differentiation to sum in quadrature and calculate error
    for s, sigma in zip(symbols, errors):
        df_ds = sp.diff(f, s)
        df_val = float(df_ds.subs(subs).evalf())
        uncertainty_squared.append(
            df_val**2 * sigma**2
        )

    # calculate values
    value = float(f.subs(subs).evalf())
    error = math.sqrt(sum(uncertainty_squared))

    return value, error