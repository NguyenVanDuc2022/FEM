import numpy as np


def EB_assemble_stiffness(E, I, Coordinate_Nodes, NoE, NoDOFs, EInd):
    Stiffness_Max = np.zeros([NoDOFs, NoDOFs])  # The global matrix of stiffness
    for i in range(NoE):
        eL = Coordinate_Nodes[i + 1] - Coordinate_Nodes[i]
        Ke = E * I / eL ** 3 * np.array([[12, 6 * eL, -12, 6 * eL],
                                         [6 * eL, 4 * eL ** 2, -6 * eL, 2 * eL ** 2],
                                         [-12, -6 * eL, 12, -6 * eL],
                                         [6 * eL, 2 * eL ** 2, -6 * eL, 4 * eL ** 2]])
        Stiffness_Max[np.array([EInd[i, :]]).transpose(), EInd[i, :]] += Ke
    return Stiffness_Max


def TB_assemble_stiffness(E, I, G, k, A, Coordinate_Nodes, NoE, NoDOFs, EInd):
    Stiffness_Max = np.zeros([NoDOFs, NoDOFs])  # The global matrix of stiffness
    for i in range(NoE):
        eL = Coordinate_Nodes[i + 1] - Coordinate_Nodes[i]
        Kb = E * I / eL * np.array([[0, 0, 0, 0],
                                    [0, 1, 0, -1],
                                    [0, 0, 0, 0],
                                    [0, -1, 0, 1]])
        Ks = k * A * G / eL * np.array([[1, eL / 2, -1, eL / 2],
                                        [eL / 2, eL ** 2 / 3, -eL / 2, eL ** 2 / 6],
                                        [-1, -eL / 2, 1, -eL / 2],
                                        [eL / 2, eL ** 2 / 6, -eL / 2, eL ** 2 / 3]])
        Ke = Kb + Ks
        Stiffness_Max[np.array([EInd[i, :]]).transpose(), EInd[i, :]] += Ke
    return Stiffness_Max


def assemble_forces(PointLoad, DistributedLoad, Coordinate_Nodes, NoE, NoDOFs, EInd):
    forces = np.zeros([NoDOFs, 1])  # The global vector of equivalent force
    for i in range(NoE):
        eL = Coordinate_Nodes[i + 1] - Coordinate_Nodes[i]
        forces_q = DistributedLoad[0] * eL * np.array([[1 / 2], [eL / 12], [1 / 2], [-eL / 12]])
        forces_m = DistributedLoad[1] * np.array([[-1], [0], [1], [0]])
        forces[EInd[i, :]] += forces_q + forces_m
    forces[0] += PointLoad[0]
    forces[1] += PointLoad[1]
    if NoE % 2 == 0:
        forces[NoE] += PointLoad[2]
        forces[NoE + 1] += PointLoad[3]
    else:
        forces[NoE - 1] += PointLoad[2] / 2
        forces[NoE] += PointLoad[3] / 2
        forces[NoE + 1] += PointLoad[2] / 2
        forces[NoE + 2] += PointLoad[3] / 2
    forces[NoDOFs - 2] += PointLoad[4]
    forces[NoDOFs - 1] += PointLoad[5]
    return forces


def assemble_displacements(NoDOFs, FixedNodes, NoDOFspN):
    u = np.zeros([NoDOFs, 1])
    Fix_NodeI = np.zeros(list(FixedNodes[:, -1]).count(0))
    for i in range(np.size(FixedNodes, 0)):
        if FixedNodes[i, 2] == 0:
            Fix_NodeI[i] = FixedNodes[i, 0] * NoDOFspN + FixedNodes[i, 1]
            u[int(Fix_NodeI[i])] = FixedNodes[i, 2]
    Free_NodeI = np.setdiff1d(np.array([range(NoDOFs)]), Fix_NodeI)
    return u, Free_NodeI, Fix_NodeI


def Beam_EulerBernoulli(E, L, Inertia, PointLoad, DistributedLoad, FixedNodes, NoE):
    NoNpE = 2  # Number of Nodes per Element
    NoDOFspN = 2  # Number of DOFs per Node
    NoN = NoE + 1  # Number of Nodes
    NoDOFspE = NoNpE * NoDOFspN  # Number of DOFs per Node
    NoDOFs = NoN * NoDOFspN  # Number of DOFs

    # Matrix of elements index
    EInd = (np.array([range(NoE)]) * NoDOFspN).transpose() + np.array(range(NoDOFspE))
    Coordinate_Nodes = np.linspace(0, L, NoN)  # Coordinates of Nodes

    # The global matrix of stiffness
    Stiffness_Max = EB_assemble_stiffness(E, Inertia, Coordinate_Nodes, NoE, NoDOFs, EInd)

    # The global vector of equivalent force
    forces = assemble_forces(PointLoad, DistributedLoad, Coordinate_Nodes, NoE, NoDOFs, EInd)

    # The global vector of nodal displacement
    (u, Free_NodeI, Fix_NodeI) = assemble_displacements(NoDOFs, FixedNodes, NoDOFspN)

    forces1 = forces - np.matmul(Stiffness_Max, u)

    u[Free_NodeI] = np.matmul(np.linalg.inv(Stiffness_Max[np.array([Free_NodeI]).transpose(),
                                                          Free_NodeI]), forces1[Free_NodeI])
    return max(u[::2, 0], key=abs)


def Beam_Timoshenko(E, L, Inertia, G, k, A, PointLoad, DistributedLoad, FixedNodes, NoE):
    NoNpE = 2  # Number of Nodes per Element
    NoDOFspN = 2  # Number of DOFs per Node
    NoN = NoE + 1  # Number of Nodes
    NoDOFspE = NoNpE * NoDOFspN  # Number of DOFs per Node
    NoDOFs = NoN * NoDOFspN  # Number of DOFs

    # Matrix of elements index
    EInd = (np.array([range(NoE)]) * NoDOFspN).transpose() + np.array(range(NoDOFspE))
    Coordinate_Nodes = np.linspace(0, L, NoN)  # Coordinates of Nodes

    # The global matrix of stiffness
    Stiffness_Max = TB_assemble_stiffness(E, Inertia, G, k, A, Coordinate_Nodes, NoE, NoDOFs, EInd)

    # The global vector of equivalent force
    forces = assemble_forces(PointLoad, DistributedLoad, Coordinate_Nodes, NoE, NoDOFs, EInd)

    # The global vector of nodal displacement
    (u, Free_NodeI, Fix_NodeI) = assemble_displacements(NoDOFs, FixedNodes, NoDOFspN)

    forces1 = forces - np.matmul(Stiffness_Max, u)

    u[Free_NodeI] = np.matmul(np.linalg.inv(Stiffness_Max[np.array([Free_NodeI]).transpose(),
                                                          Free_NodeI]), forces1[Free_NodeI])
    return max(u[::2, 0], key=abs)
