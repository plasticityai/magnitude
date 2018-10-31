u"""
An :class:`Activation` is just a function
that takes some parameters and returns an element-wise activation function.
For the most part we just use
`PyTorch activations <http://pytorch.org/docs/master/nn.html#non-linear-activations>`_.
Here we provide a thin wrapper to allow registering them and instantiating them ``from_params``.

The available activation functions are

* "linear"
* `"relu" <http://pytorch.org/docs/master/nn.html#torch.nn.ReLU>`_
* `"relu6" <http://pytorch.org/docs/master/nn.html#torch.nn.ReLU6>`_
* `"elu" <http://pytorch.org/docs/master/nn.html#torch.nn.ELU>`_
* `"prelu" <http://pytorch.org/docs/master/nn.html#torch.nn.PReLU>`_
* `"leaky_relu" <http://pytorch.org/docs/master/nn.html#torch.nn.LeakyReLU>`_
* `"threshold" <http://pytorch.org/docs/master/nn.html#torch.nn.Threshold>`_
* `"hardtanh" <http://pytorch.org/docs/master/nn.html#torch.nn.Hardtanh>`_
* `"sigmoid" <http://pytorch.org/docs/master/nn.html#torch.nn.Sigmoid>`_
* `"tanh" <http://pytorch.org/docs/master/nn.html#torch.nn.Tanh>`_
* `"log_sigmoid" <http://pytorch.org/docs/master/nn.html#torch.nn.LogSigmoid>`_
* `"softplus" <http://pytorch.org/docs/master/nn.html#torch.nn.Softplus>`_
* `"softshrink" <http://pytorch.org/docs/master/nn.html#torch.nn.Softshrink>`_
* `"softsign" <http://pytorch.org/docs/master/nn.html#torch.nn.Softsign>`_
* `"tanhshrink" <http://pytorch.org/docs/master/nn.html#torch.nn.Tanhshrink>`_
"""


from __future__ import absolute_import
import torch

from allennlp.common import Registrable


class Activation(Registrable):
    u"""
    Pytorch has a number of built-in activation functions.  We group those here under a common
    type, just to make it easier to configure and instantiate them ``from_params`` using
    ``Registrable``.

    Note that we're only including element-wise activation functions in this list.  You really need
    to think about masking when you do a softmax or other similar activation function, so it
    requires a different API.
    """
    def __call__(self, tensor              )                :
        u"""
        This function is here just to make mypy happy.  We expect activation functions to follow
        this API; the builtin pytorch activation functions follow this just fine, even though they
        don't subclass ``Activation``.  We're just making it explicit here, so mypy knows that
        activations are callable like this.
        """
        raise NotImplementedError

# There are no classes to decorate, so we hack these into Registrable._registry
# pylint: disable=protected-access
Registrable._registry[Activation] = {  # type: ignore
        u"linear": lambda: lambda x: x,
        u"relu": torch.nn.ReLU,
        u"relu6": torch.nn.ReLU6,
        u"elu": torch.nn.ELU,
        u"prelu": torch.nn.PReLU,
        u"leaky_relu": torch.nn.LeakyReLU,
        u"threshold": torch.nn.Threshold,
        u"hardtanh": torch.nn.Hardtanh,
        u"sigmoid": torch.nn.Sigmoid,
        u"tanh": torch.nn.Tanh,
        u"log_sigmoid": torch.nn.LogSigmoid,
        u"softplus": torch.nn.Softplus,
        u"softshrink": torch.nn.Softshrink,
        u"softsign": torch.nn.Softsign,
        u"tanhshrink": torch.nn.Tanhshrink,
}
