import numpy as np

def remainder(i,n,d):

    """Write i = i_1*(n+1)^{d-1} + ... + i_d*(n+1)^{0} as the (n+1)-ary
    representation of i, given n and d are also known. The remainder()
    function returns the list [i_1, ..., i_d]."""

    #import numpy as np

    index_vec = np.zeros(d)
    i_temp = i

    for j in range(0,d):

        temp = divmod(i_temp, (n+1)**(d-1-j))
        index_vec[j] = temp[0]
        i_temp = temp[1]
        j += 1

    index_vec = index_vec.astype(int)
    return index_vec

class Polynomial(object):

    def __init__(self, coeffs):

        """For a polynomial in dval variables with nval as the greatest exponent
        of any single variable, coeffs should be an array with shape
        (nval + 1,...,nval + 1), with repetition dval times.

        The coefficient of x_1^{i_1}*...*x_dval^{i_dval} will then be
        coeffs[i_1,...,i_dval].
        """

        self.coeffs = coeffs
        self.dval = len(coeffs.shape)
        self.nval = coeffs.shape[0] - 1


    def evalAt(self, X):

        """Evaluates self at a list X = [x_1, ..., x_dval]."""
        evalVec = self.coeffs.ravel()
        for i in np.nonzero(evalVec)[0].tolist(): # Ignore terms with coeff = 0.

            for j in range(0,self.dval):

                """This loop calculates
                coeff[i_1, ..., i_d] * X[0]^(i_1) * ... * X[d-1]^(i_d)."""

                evalVec[i] = \
                evalVec[i] * (X[j] ** \
                remainder(i, self.nval, self.dval).tolist()[j])

        return sum(evalVec) # Sum the terms to get result.

    def compare(self, other):

        """Takes two polynomials, self and other, and returns
        two new coeff arrays of the same shape, corresponding to the same
        polynomials as self.coeffs, other.coeffs, respectively."""

        newD = np.max(np.array[self.dval, other.dval])
        newN = np.max(np.array[self.nval, other.nval]) 
        # Find shape dimensions for new array.

        newSelfRavel = np.zeros((newN+1) ** newD)
        newOtherRavel = np.zeros((newN+1) ** newD)

        for i in np.nonzero(self.coeffs.ravel())[0].tolist():

            iNew = 0
            for j in range(0, self.dval):

                iNew += \
                remainder(i, self.nval, self.dval).tolist()[j] * \
                ((newN +1) ** (newD -j -1))

            newSelfRavel[iNew] = self.coeffs.ravel()[i]

        for k in np.nonzero(other.coeffs.ravel())[0].tolist():

            iNew2 = 0
            for h in range(0, other.dval):

                iNew2 += \
                remainder(k, other.nval, other.dval).tolist()[h] * \
                ((newN +1) ** (newD -h -1))

            newOtherRavel[iNew2] = other.coeffs.ravel()[k]

        return [newSelfRavel, newOtherRavel]


    def addP(self, other):

        ravelSum = sum(self.compare(other))
        newShape = []
        for i in range(np.max(np.array[self.dval, other.dval])):
            newShape.append(np.max(np.array[self.nval, other.nval]) +1)

        newShape = tuple(newShape)
        coeffSum = ravelSum.reshape(newShape)
        return Polynomial(coeffSum)


    def mulP(self,other):

        mulDval = np.max(np.array[self.dval, other.dval])
        mulNval = self.nval + other.nval

        ravelMul = np.zeros((mulNval +1) ** mulDval) # Base for multn.

        for i in np.nonzero(self.coeffs.ravel())[0].tolist():
            for j in np.nonzero(other.coeffs.ravel())[0].tolist():

                mulCoeff = self.coeffs.ravel()[i] * other.coeffs.ravel()[j]

                iLoc = np.asarray(remainder(i, self.nval, self.dval))
                iLocPad = np.lib.pad(iLoc, (0, mulDval -self.dval), 'constant')

                jLoc = np.asarray(remainder(j, other.nval, other.dval))
                jLocPad = np.lib.pad(jLoc, (0, mulDval -other.dval), 'constant')

                mulIndexVec = iLocPad + jLocPad
                mulIndexVec = mulIndexVec.tolist()
                mulIndex = 0

                for k in range(0, mulDval):
                    mulIndex += mulIndexVec[k] * \
                            ((mulNval +1) ** (mulDval -k -1))

                ravelMul[mulIndex] += mulCoeff

        mulShape = []
        for i in range(mulDval):
            mulShape.append(mulNval +1)

        mulShape = tuple(mulShape)
        coeffMul = ravelMul.reshape(mulShape)
        return Polynomial(coeffMul)
