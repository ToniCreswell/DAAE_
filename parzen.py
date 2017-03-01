import argparse
import time
import gc
import numpy
import theano
import theano.tensor as T



def get_nll(x, parzen, batch_size=10):
    """
    Credit: Yann N. Dauphin
    """

    inds = range(x.shape[0])
    n_batches = int(numpy.ceil(float(len(inds)) / batch_size))

    times = []
    nlls = []
    for i in range(n_batches):
        begin = time.time()
        nll = parzen(x[inds[i::n_batches]])
        end = time.time()
        times.append(end-begin)
        nlls.extend(nll)

        if i % 100 == 0:
            print i, numpy.mean(times), numpy.mean(nlls)

    return numpy.array(nlls)


def log_mean_exp(a):
    """
    Credit: Yann N. Dauphin
    """

    max_ = a.max(1)

    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def theano_parzen(mu, sigma):
    """
    Credit: Yann N. Dauphin
    """

    x = T.matrix()
    mu = theano.shared(mu)
    a = ( x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1) ) / sigma
    E = log_mean_exp(-0.5*(a**2).sum(2))
    Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))

    return theano.function([x], E - Z)


def cross_validate_sigma(samples, train, sigmas, batch_size):

    lls = []
    for sigma in sigmas:
        print sigma
        parzen = theano_parzen(samples, sigma)
        tmp = get_nll(train, parzen, batch_size = batch_size)
        lls.append(numpy.asarray(tmp).mean())
        del parzen
        gc.collect()

    ind = numpy.argmax(lls)
    return sigmas[ind]

    # fit and evaulate
    parzen = theano_parzen(samples, sigma)
    ll = get_nll(test, parzen, batch_size = batch_size)
    se = ll.std() / numpy.sqrt(test.X.shape[0])

    print "Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se)

    # valid
    ll = get_nll(valid, parzen, batch_size = batch_size)
    se = ll.std() / numpy.sqrt(val.shape[0])
    print "Log-Likelihood of valid set = {}, se: {}".format(ll.mean(), se)


if __name__ == "__main__":
    main()





