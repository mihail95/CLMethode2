from vecmat.entities import Vector
from vecmat.vmtypes import VectorType

def run_script():

    # Test initialization
    print("\nINITIALIZE:\n-----------------")
    vector1 = Vector([1,2,3])
    vector2 = Vector([4,5,6], VectorType.ROW)

    vector1.show()
    vector2.show()

    # Test transposition
    print("\n TRANSPOSE:\n-----------------")
    vector1.transpose()
    vector2.transpose()
    vector1.show()
    vector2.show()

    # Test operations
    print("\n OPERATIONS:\n-----------------")
    vector2.transpose()
    vector3 = vector1.add(vector2)
    vector3.show()
    print("Dimensions check: ", (vector1.dimCol == vector3.dimCol) and (vector1.dimRow == vector3.dimRow))

###########################################################
# Hauptprogramm
###########################################################

if __name__ == "__main__":

    # Funktion, die alle weiteren Funktionen aufruft
    run_script()