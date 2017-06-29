# multinomial-classes
This is an implementation of two different classes of multinomials: commutative and non-commutative. In these classes, one is able to add and multiply polynomials, and evaluate them at a point in R^d.

HOW TO INSTANTIATE A COMMUTATIVE POLYNOMIAL:
For a polynomial in dval variables with nval as the greatest exponent
        of any single variable, coeffs should be a numpy array with shape
        (nval + 1,...,nval + 1), with repetition dval times.

        The coefficient of x_1^{i_1}*...*x_dval^{i_dval} will then be
        coeffs[i_1,...,i_dval]. Then Polynomial(coeffs) will be a 
        polynomial with the desired coefficients.
