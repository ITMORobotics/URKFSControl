import numpy as np
import sys,os
import PyKDL as kdl

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kdl_parser.kdl_parser_py.kdl_parser_py import urdf

np.set_printoptions(precision=4)

def main():

    q = kdl.JntArray(6)
    print(q)
    base_link = 'world'
    tool_link = 'tool0'
    urdf_file = open('urdf_model/ur5e.urdf','r')
    urdf_str = urdf_file.read()
    urdf_file.close()

    # Generate kinematic model for orocos_kdl
    (ok, tree) = urdf.treeFromString(urdf_str)
    chain = tree.getChain(base_link, tool_link)

    jac_solver = kdl.ChainJntToJacSolver(chain)
    jac = kdl.Jacobian(6)
    jac_solver.JntToJac(q, jac)
    print(jac)

if __name__ == "__main__":
    main()