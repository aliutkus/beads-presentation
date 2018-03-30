import numpy as np
from scipy.special import erf
from scipy.stats import norm
from matplotlib import cm
import numbers
from copy import copy
from itertools import product
from matplotlib import pylab as plt


def cgauss(x, mu, sigma):
    return np.real(1./np.pi/sigma*np.exp(-np.abs(x-mu)**2 / sigma))


def vec(z):
    # makes a bidimensional array out of complex z
    z = np.array(z)
    return np.concatenate((np.real(z[None, ...]), np.imag(z[None, ...])))


def w(n):
    return np.exp(1j*2*np.pi/n)


class Distribution(object):
    """Distribution:

    base class for the probabilistic models. Implements just the basics
    about the weight of this object, for convenience in the mixture models."""
    def __init__(self):
        # initialize the weight to 1
        self.weight = 1

    def pdf(self, z):
        # will compute the pdf. To be overridden
        pass

    def __rmul__(self, other):
        # multiplying left-wise by a scalar means modifying the weight
        if isinstance(other, numbers.Number):
            result = copy(self)
            result.weight *= other
            return result
        elif other is None:
            return self
        else:
            raise ArithmeticError('Cannot left multiply a distribution '
                                  'by anything else than a number, for '
                                  'the purpose of assigning a weight.')

    def plot(self, canvas, ax=None, colors='Reds', nlines=100, alpha=None):
        if ax is None:
            ax = plt.gca()
        density = self.pdf(canvas.Z)

        cset = ax.contour(canvas.X, canvas.Y, density,
                          levels=np.linspace(density.min(),
                                             density.max(), nlines),
                          cmap=getattr(cm, colors), alpha=alpha)
        return cset


class Bead(Distribution):
    """Bead:

    A Bead object is a simple complex isotropic Gaussian."""
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.references = []

    def pdf(self, z):
        return self.weight * cgauss(z, self.mu, self.sigma)

    def __copy__(self):
        result = Bead(self.mu, self.sigma)
        result.weight = self.weight
        return result

    @staticmethod
    def fromBeads(references):
        mean = sum([ref.mu for ref in references])
        sigma = sum([ref.sigma for ref in references])
        result = Bead(mean, sigma)
        result.references = references
        return result


class Donut(Distribution):
    """Donut:

    The Donut class implements the ideal distribution that the BEADS
    model approximates."""

    def __init__(self, mu, b, sigma):
        super().__init__()
        self.mu = mu
        self.b = b
        self.sigma = sigma/2

    def pdf(self, z):
        radius = np.abs(z-self.mu)
        F = norm(self.b, self.sigma)
        # the normalizing constant for the donut distribution is
        # \int_{r,\theta}f\left(r\mid b,\sigma\right)drrd\theta
        # =\pi b(1-erf(-\frac{b}{\sigma\sqrt{2}}))
        # +\sqrt{2\pi}\exp(-\frac{b^{2}}{2\sigma2})$
        Z = (np.pi * self.b * (1-erf(-self.b / self.sigma / np.sqrt(2)))
             + np.sqrt(2 * np.pi)*np.exp(-self.b**2/2/self.sigma**2))
        return self.weight / Z * F.pdf(radius)

    def __copy__(self):
        result = Donut(self.mu, self.b, self.sigma)
        result.weight = self.weight
        return result


class GMM(Distribution):
    """GMM:

    A Gaussian Mixture Model is a collection of Bead objects. The objects are
    in arbitrary numbers and positions, with arbitrary weights."""

    def __init__(self):
        super().__init__()
        self.components = []
        self.product_of = []

    def total_weight(self):
        return self.weight * sum([comp.weight for comp in self.components])

    def __iadd__(self, other):
        if not (isinstance(other, Bead) or isinstance(other, GMM)):
            raise ArithmeticError('can only add a GMM or a Bead to a GMM')

        if isinstance(other, Bead):
            # if we want to add a Bead, we simply append it to the components
            self.components += [other]
            return self

        # more complicated case: we add one GMM to another. In this case,
        # we need to weight the components of each according to each GMM
        # global weight, and doing so, we make new copies of the Bead objects.

        # the multiplication creates new copies
        self.components = [self.weight * comp for comp in self.components]
        other_components = [other.weight * comp for comp in other.components]
        self.components += other_components
        total_weight = sum([comp.weight for comp in self.components])
        for comp in self.components:
            comp.weight /= total_weight
        self.weight = total_weight
        return self

    def __add__(self, other):
        result = copy(self)
        result += other
        return result

    def pdf(self, z):
        result = np.zeros(z.shape)
        for component in self.components:
            result += component.pdf(z)
        return result

    def __copy__(self):
        result = GMM()
        result.weight = self.weight
        result.components = [comp for comp in self.components]
        return result

    def __mul__(self, other):
        if other is None:
            return self

        if not isinstance(other, (GMM, Beads)):
            raise ArithmeticError('Can only multiply GMM with GMM')
        if other in self.product_of:
            raise ArithmeticError('Cannot include twice the same GMM in a'
                                  'product. Another one is needed.')

        result = GMM()

        # we want the product not to be nested: all operands need to be
        # simple GMM and not already part of the product
        def flatten(a, b, attr):
            attr_a = getattr(a, attr)
            attr_b = getattr(b, attr)
            res = attr_a + attr_b
            if not len(attr_a):
                res += [a]
            if not len(attr_b):
                res += [b]
            return res
        result.product_of = flatten(self, other, "product_of")

        # now incorporate all couples in the product components
        for a, b in product(self.components, other.components):
            references = flatten(a, b, "references")
            result += Bead.fromBeads(references)
        return result

    def post(self, mix, x):
        # if self is independent from mix, just return a copy
        if self not in mix.product_of:
            return copy(self)

        result = GMM()
        if all([isinstance(x, Beads) for x in mix.product_of]):
            # the case where the mix is a sum of Beads objects
            total_weight = 0
            for xcomp in mix.components:
                scomp = [c for c in self.components if c in xcomp.references]
                if len(scomp) != 1:
                    raise IndexError('One mix component featured no unique'
                                     'Bead object from the source as'
                                     'reference')
                scomp = scomp[0]
                sigmas = scomp.sigma
                sigmax = xcomp.sigma
                G = sigmas/sigmax

                pi_post = scomp.weight * xcomp.pdf(x)
                result += pi_post * Bead(scomp.mu+G*(x-xcomp.mu),
                                         sigmas*(1-G))
                total_weight += pi_post
            for comp in result.components:
                comp.weight /= total_weight
            return result
        else:
            raise NotImplemented


class Beads(GMM):
    def __init__(self, mu, b, sigma, weights):
        super().__init__()
        if isinstance(weights, int):
            weights = np.ones(weights)
        weights /= weights.sum()
        n = len(weights)
        omega = mu + b * w(n)**np.arange(0, n, 1)
        for (center_c, pi_c) in zip(omega, weights):
            self += pi_c * Bead(center_c, sigma)


class Canvas:
    def __init__(self, min, max, N):
        self.N = N
        self.min = min
        self.max = max

        # create a meshgrid
        X = np.linspace(min, max, N)
        Y = np.linspace(min, max, N)
        self.X, self.Y = np.meshgrid(X, Y)
        self.Z = self.X + 1j*self.Y

        # create the canvas of given dimensions
        self.clear()

    def clear(self):
        # Create canvas
        self.canvas = np.zeros((self.N, self.N))

    def __add__(self, other):
        if isinstance(other, Distribution):
            self.canvas += other.pdf(self.Z)
        elif isinstance(other, Canvas):
            self.canvas += other.canvas
        else:
            raise ArithmeticError('Can only add Distribution or Canvas '
                                  'objects (of the same size) to Canvas.')
        return self

    def plot(self, ax=None, colors='Reds', nlines=100):
        if ax is None:
            ax = plt.gca()
        cset = ax.contour(self.X, self.Y, self.canvas,
                          levels=np.linspace(self.canvas.min(),
                                             self.canvas.max(), nlines),
                          cmap=getattr(cm, colors))
        return cset
