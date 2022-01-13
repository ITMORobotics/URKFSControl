import numpy as np
import dartpy as dart

np.set_printoptions(precision=4)

def main():

    urdfParser = dart.utils.DartLoader()
    kr5 = urdfParser.parseSkeleton("/media/files/URKFSControl/urdf_model/ur5e.urdf")
    
    dofs = kr5.getNumDofs()
    ee = kr5.getBodyNode('tool0')
    J = ee.getWorldJacobian([0.0,0.0,0.0])
    twist = ee.getTwist()
    print("Full jacobian:")
    print(np.array(J))
    print("LinearJac:")
    print(ee.getLinearJacobian())
    print("AngularJac:")
    print(ee.getAngularJacobian())
        
    kr5.setPositions([0.0, np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
    print("Actual q")
    print(kr5.getPositions())

    J = ee.getJacobian([0.0,0.0,0.0])
    print(J)

if __name__ == "__main__":
    main()