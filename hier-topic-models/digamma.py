import numpy as np

# Approximating digamma (psi)
# Source: https://en.wikipedia.org/wiki/Digamma_function#Computation_and_approximation
def psi(x):
    x = np.asarray(x).astype('float')
    input_shape = x.shape
    x = np.atleast_1d(x)
    result = np.zeros(x.shape)
    # First, use the recurrence relation psi(x) = psi(x+1) - 1.0/x
    # to express psi(x) in terms of psi(x+n), where x+n >= 6.
    while any(x <= 6):
        result -= 1.0 / x
        x += 1
    # Then, use a series expansion which is valid for sufficiently large x
    result += (
        np.log(x)
        - 1/2. * x**-1
        - 1/12. * x**-2
        + 1/120. * x**-4
        - 1/252. * x**-6
        + 1/240. * x**-8
        - 5/660. * x**-10
        + 691/32760. * x**-12
        - 1/12. * x**-14)
    if len(input_shape) == 0:
        return np.asscalar(result)
    return result.reshape(input_shape)

'''
For octave/matlab:

% Implementation of digamma (psi) is missing in some versions of Octave
% Source: https://en.wikipedia.org/wiki/Digamma_function#Computation_and_approximation
function r = psi(x)
    r = 0;
    % First, use the recurrence relation psi(x) = psi(x+1) - 1.0/x
    % to express psi(x) in terms of psi(x+n), where x+n >= 6.
    while x <= 6
        r = r - 1./x;
        x = x + 1;
    end
    % Then, use a series expansion which is valid for sufficiently large x
    r = r + log(x) - 1/2*x.^-1 - 1/12*x.^-2 + 1/120*x.^-4 ...
        - 1/252*x.^-6 + 1/240*x.^-8 - 5/660*x.^-10 ...
        + 691/32760*x.^-12 - 1/12*x.^-14;
endfunction
'''

if __name__ == "__main__":
    em_constant = 0.57721566490153
    print("Euler-Mascheroni constant: {}".format(em_constant))
    print("psi(1): {}".format(psi(1)))
    print("psi(2): {}".format(psi(2)))
    print("psi(3): {}".format(psi(3)))
    print("psi([1,2,1,3,1]): {}".format(psi([1,2,1,3,1])))
