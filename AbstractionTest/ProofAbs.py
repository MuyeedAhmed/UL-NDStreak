# gurobi==11.x
from gurobipy import Model, GRB, quicksum

def build_ap_light_two_runs(n=4, d=2, eps=1e-3, LB=-1.0, UB=1.0, bigM=None, time_limit=300):
    m = Model("AP_light_two_runs_max_disagree")


    # X = m.addVars(n, d, lb=LB, max=UB, name="X")
    X = m.addVars(n, d, vtype=GRB.CONTINUOUS, name="X")

    dij = m.addVars(n, n, lb=0.0, vtype=GRB.CONTINUOUS, name="d")

    p = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="p")

    r1 = m.addVars(n, n, lb=-eps, ub=eps, name="r1")
    r2 = m.addVars(n, n, lb=-eps, ub=eps, name="r2")

    # Argmin selection binaries for each run
    y1 = m.addVars(n, n, vtype=GRB.BINARY, name="y1")  # y1[i,j]=1 if run1 picks j for row i
    y2 = m.addVars(n, n, vtype=GRB.BINARY, name="y2")

    # Same-choice indicator per i (for objective)
    w  = m.addVars(n, n, vtype=GRB.BINARY, name="w")   # w[i,j] ≤ y1[i,j], w[i,j] ≤ y2[i,j]
    same = m.addVars(n, vtype=GRB.BINARY, name="same") # same[i] = sum_j w[i,j]
    # disagreements per row = 1 - same[i]

    # ---------- Helper constants ----------
    # Crude but safe upper bound for any squared distance using the box:
    # max per-dim delta = (UB-LB), so ||xi-xj||^2 ≤ d*(UB-LB)^2
    DUB = d * (UB - LB) ** 2
    for i in range(n):
        for j in range(n):
            dij[i, j].UB = DUB

    # Big-M for argmin ordering (≥ maximum possible spread after noise and diag swap)
    if bigM is None:
        # base bound plus diag change plus noise
        bigM = DUB + abs(DUB) + 2*eps + 1.0  # +1 margin

    # ---------- Constraints ----------
    # Distances as quadratic lower bounds (MIQCP)
    for i in range(n):
        for j in range(n):
            if i == j:
                # leave dij[i,i] free (we won't use it for the median); keep nonneg LB
                continue
            expr = quicksum((X[i, k] - X[j, k]) * (X[i, k] - X[j, k]) for k in range(d))
            m.addQConstr(dij[i, j] >= expr, name=f"sqdist_lb[{i},{j}]")

    # Label selection — each row picks exactly one column in each run
    for i in range(n):
        m.addConstr(quicksum(y1[i, j] for j in range(n)) == 1, name=f"pick1[{i}]")
        m.addConstr(quicksum(y2[i, j] for j in range(n)) == 1, name=f"pick2[{i}]")

    # Build distance entries AFTER median-on-diag and noise:
    # For run r ∈ {1,2}:  D^r[i,j] = (i==j ? p : d_ij) + r^r[i,j]
    # Encode argmin with big-M: if y[i,j]=1 then D[i,k] ≥ D[i,j] for all k
    def add_argmin_constraints(y, r):
        for i in range(n):
            for j in range(n):
                Dj = (p if i == j else dij[i, j]) + r[i, j]
                for k in range(n):
                    Dk = (p if i == k else dij[i, k]) + r[i, k]
                    # Dk >= Dj - M*(1 - y[i,j])
                    m.addConstr(Dk >= Dj - bigM*(1 - y[i, j]),
                                name=f"argmin_ord[i{i}]_j{j}_k{k}_{y._name}")
    add_argmin_constraints(y1, r1)
    add_argmin_constraints(y2, r2)

    # --- Median constraints for p over OFF-DIAGONAL distances only ---
    # Use counting with binaries a_ij (≤ p) and b_ij (≥ p)
    a = m.addVars(n, n, vtype=GRB.BINARY, name="a_le_p")  # a=1 ⇒ d_ij ≤ p
    b = m.addVars(n, n, vtype=GRB.BINARY, name="b_ge_p")  # b=1 ⇒ d_ij ≥ p

    Noff = n*n - n
    half = (Noff + 1)//2  # ceil(Noff/2)

    # linearize with big-M around dij and p (these are linear here)
    for i in range(n):
        for j in range(n):
            if i == j:
                # We don't use diagonal entries in the median count
                m.addConstr(a[i, j] == 0)
                m.addConstr(b[i, j] == 0)
                continue
            # a=1 -> d_ij ≤ p
            m.addConstr(dij[i, j] - p <=  bigM*(1 - a[i, j]), name=f"d_le_p[{i},{j}]")
            # b=1 -> d_ij ≥ p
            m.addConstr(p - dij[i, j] <= bigM*(1 - b[i, j]), name=f"d_ge_p[{i},{j}]")

    # At least half distances ≤ p and at least half ≥ p
    m.addConstr(quicksum(a[i, j] for i in range(n) for j in range(n)) >= half,
                name="median_lower_half")
    m.addConstr(quicksum(b[i, j] for i in range(n) for j in range(n)) >= half,
                name="median_upper_half")

    # --- Same/different bookkeeping for the objective ---
    for i in range(n):
        # w[i,j] ≤ y1[i,j], w[i,j] ≤ y2[i,j]
        for j in range(n):
            m.addConstr(w[i, j] <= y1[i, j])
            m.addConstr(w[i, j] <= y2[i, j])
        # same[i] == sum_j w[i,j]  (since y1 and y2 each pick exactly one j)
        m.addConstr(same[i] == quicksum(w[i, j] for j in range(n)))

    # ---------- Objective ----------
    # Maximize disagreements = sum_i (1 - same[i])
    disagree = quicksum(1 - same[i] for i in range(n))
    m.setObjective(disagree, GRB.MAXIMIZE)

    # ---------- Hints / params ----------
    m.Params.OutputFlag = 1
    if time_limit:
        m.Params.TimeLimit = time_limit

    return m, dict(X=X, dij=dij, p=p, r1=r1, r2=r2, y1=y1, y2=y2, same=same)

# Example run (tweak n,d,eps as needed)
if __name__ == "__main__":
    n, d = 8, 2
    eps = 1e-3
    model, vars = build_ap_light_two_runs(n=n, d=d, eps=eps, LB=-1.0, UB=1.0, time_limit=300)
    model.optimize()

    if model.SolCount > 0:
        # Extract how many rows differ
        same = sum(int(vars["same"][i].X + 0.5) for i in range(n))
        print(f"Rows with different argmin: {n - same} / {n}")
