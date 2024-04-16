from .vmtypes import VectorType

class Vector:
    """Initialize a Vector 
    @Input Params:
        data (array): An array of values
        type (VectorType): Row or Column (default) vector
    """
    def __init__(self, data:list, type:VectorType = VectorType.COLUMN):
        self.values = data
        self.type = type
        self.dimRow = 1 if type == VectorType.ROW else len(data)
        self.dimCol = 1 if type == VectorType.COLUMN else len(data)

    def transpose(self):
        """Turns a column vector into a row vector and vice versa"""
        if self.type == VectorType.ROW:
            self.type = VectorType.COLUMN
            self.dimRow = self.dimCol
            self.dimCol = 1
        else:
            self.type = VectorType.ROW
            self.dimCol = self.dimRow
            self.dimRow = 1

    def show(self):
        """Prints the vector in the right direction (hopefully)"""
        for row in range(self.dimRow):
            print("[", end=" ")
            for col in range(self.dimCol):
                if self.type == VectorType.ROW: print(f"{self.values[col]}", end=" ") 
                elif self.type == VectorType.COLUMN: print(f"{self.values[row]}", end=" ")
            print("]")
    
    def add(self, target:'Vector'):
        """Adds all values from the current vector to the target vector values.
        The two vectors must be of the same type and have the same dimensions
        
        @Input Params:
            target (Vector): Target vector

        @Output:
            result (Vector): A new vector of the same type and dimensions"""

        self.checkSameType(target)
        self.checkSameDimensions(target)
        	
        resultArr = [value + target.values[key] for key, value in enumerate(self.values)]        
        return Vector(resultArr, self.type)

    def dot(self, target:'Vector'):
        """Calculates the dot product of all values from the current vector to the target vector values.
        The two vectors must be of the same type and have the same dimensions
        
        @Input Params:
            target (Vector): Target vector

        @Output:
            result (int): The result scalar"""
        
        self.checkSameType(target)
        self.checkSameDimensions(target)

        result = sum(map(lambda x,y: x*y, self.values, target.value))
        return result
        
    def checkSameType(self, target:'Vector'):
        if self.type != target.type:
            raise TypeError("Only vectors of the same type can be added together! Try transposing.")
        
    def checkSameDimensions(self, target:'Vector'):
        if (self.dimCol != target.dimCol) or (self.dimRow != target.dimRow):
            raise TypeError("Only vectors of the same dimensions can be added together!")
