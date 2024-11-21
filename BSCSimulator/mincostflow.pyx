from ortools.graph import pywrapgraph
import numpy as np
cimport numpy as cnp

cnp.import_array()


def mincostflow(cnp.ndarray supply, cnp.ndarray demand, cnp.ndarray cost, cnp.ndarray scaled_cost):
    cdef int n = supply.size
    cdef int m = demand.size
    cdef int tot = max(supply.sum(), demand.sum())
    cdef cnp.ndarray pens = cost < 1e17
    cdef int pen_sum = pens.sum()
    cdef int[:,:] true_penalty = (pens * 1).astype(np.intc)
    scaled_cost = scaled_cost * 1000
    return _mincostflow(supply.astype(np.intc), demand.astype(np.intc), cost, scaled_cost.astype(int), n, m, tot, true_penalty, pen_sum)


def maxflow_mincost(cnp.ndarray supply, cnp.ndarray demand, cnp.ndarray cost, cnp.ndarray scaled_cost):
    cdef int n = supply.size
    cdef int m = demand.size
    cdef int tot = max(supply.sum(), demand.sum())
    cdef cnp.ndarray pens = cost < 1e17
    cdef int pen_sum = pens.sum()
    cdef int[:,:] true_penalty = (pens * 1).astype(np.intc)
    scaled_cost = scaled_cost * 1000
    return _maxflow(supply.astype(np.intc), demand.astype(np.intc), cost, scaled_cost.astype(int), n, m, tot, true_penalty, pen_sum)


cdef cnp.ndarray _mincostflow(int[:] supply, int[:] demand, double[:, :] cost, long[:,:] scaled_cost, int n, int m, int tot, int[:,:] true_penalty, int pen_sum):
    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow(reserve_num_nodes=n+m, reserve_num_arcs=pen_sum)
    # Add each arc.
    cdef int i
    cdef int j
    cdef int[:,:] arcs = np.ones((n, m), dtype=np.intc) * -1
    for i in range(n):
        for j in range(m):
            if true_penalty[i, j] > 0:
                arcs[i, j] = min_cost_flow.AddArcWithCapacityAndUnitCost(
                    i, n+j, tot, scaled_cost[i, j])
    # Add node supplies.
    for i in range(n):
        min_cost_flow.SetNodeSupply(i, supply[i])
    for j in range(m):
        min_cost_flow.SetNodeSupply(j+n, -demand[j])
    # Find the minimum cost flow
    cdef int result = min_cost_flow.Solve()
    # Retrieve the optimal flows
    cdef cnp.ndarray[int, ndim=2] plan = np.zeros((n, m), dtype=np.intc)
    cdef int[:,:] plan_view = plan
    if result == min_cost_flow.OPTIMAL:
        for i in range(n):
            for j in range(m):
                if arcs[i, j] >= 0:
                    plan_view[i, j] = min_cost_flow.Flow(arcs[i, j])
        return plan
    else:
        print(f'Problems!!! result = {result}')
        raise RuntimeError('There was an issue with the mincostflow solver.')


cdef cnp.ndarray _maxflow(int[:] supply, int[:] demand, double[:, :] cost, long[:,:] scaled_cost, int n, int m, int tot, int[:,:] true_penalty, int pen_sum):
    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow(reserve_num_nodes=n+m, reserve_num_arcs=pen_sum)
    # Add each arc.
    cdef int i
    cdef int j
    cdef int[:,:] arcs = np.ones((n, m), dtype=np.intc) * -1
    for i in range(n):
        for j in range(m):
            if true_penalty[i, j] > 0:
                arcs[i, j] = min_cost_flow.AddArcWithCapacityAndUnitCost(
                    i, n+j, tot, scaled_cost[i, j])
    # Add node supplies.
    for i in range(n):
        min_cost_flow.SetNodeSupply(i, supply[i])
    for j in range(m):
        min_cost_flow.SetNodeSupply(j+n, -demand[j])
    # Find the minimum cost flow
    cdef int result = min_cost_flow.SolveMaxFlowWithMinCost()
    # Retrieve the optimal flows
    cdef cnp.ndarray[int, ndim=2] plan = np.zeros((n, m), dtype=np.intc)
    cdef int[:,:] plan_view = plan
    if result == min_cost_flow.OPTIMAL:
        for i in range(n):
            for j in range(m):
                if arcs[i, j] >= 0:
                    plan_view[i, j] = min_cost_flow.Flow(arcs[i, j])
        return plan
    else:
        print(f'Problems!!! result = {result}')
        raise RuntimeError('There was an issue with the mincostflow solver.')