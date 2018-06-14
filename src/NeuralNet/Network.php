<?php

namespace Rubix\ML\NeuralNet;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Parametric;
use InvalidArgumentException;
use RuntimeException;
use ArrayAccess;
use Countable;

class Network implements ArrayAccess, Countable
{
    /**
     * The layers of the network.
     *
     * @var array
     */
    protected $layers = [
        //
    ];

    /**
     * @param  \Rubix\ML\NeuralNet\Layers\Input  $input
     * @param  array  $hidden
     * @param  \Rubix\ML\NeuralNet\Layers\Output  $output
     * @return void
     */
    public function __construct(Input $input, array $hidden, Output $output)
    {
        $this->layers[] = $input;

        foreach ($hidden as $layer) {
            if (!$layer instanceof Hidden) {
                throw new InvalidArgumentException('Cannot use '
                    . get_class($layer) . ' as a hidden layer.');
            }

            $this->layers[] = $layer;
        }

        $this->layers[] = $output;
    }

    /**
     * Return the input layer.
     *
     * @return array
     */
    public function input() : Input
    {
        return $this->layers[0];
    }

    /**
     * Return an array of hidden layers indexed left to right.
     *
     * @return array
     */
    public function hidden() : array
    {
        return array_slice($this->layers, 1, count($this->layers) - 2, true);
    }

    /**
     * Return the output layer.
     *
     * @return array
     */
    public function output() : Output
    {
        return $this->layers[count($this->layers) - 1];
    }

    /**
     * The parametric layers of the network. i.e. the layers that have weights.
     *
     * @return array
     */
    public function parametric() : array
    {
        return array_filter($this->layers, function ($layer) {
            return $layer instanceof Parametric;
        });
    }

    /**
     * The depth of the network. i.e. the number of layers.
     *
     * @return int
     */
    public function depth() : int
    {
        return count($this->layers);
    }

    /**
     * Initialize the network.
     *
     * @return self
     */
    public function initialize() : self
    {
        for ($i = 1; $i < count($this->layers); $i++) {
            $current = $this->layers[$i];
            $previous = $this->layers[$i - 1];

            $current->initialize($previous);
        }

        return $this;
    }

    /**
     * Feed a sample through the network and return the output activations
     * of each output neuron.
     *
     * @param  array  $samples
     * @return self
     */
    public function feed(array $samples) : self
    {
        $input = MatrixFactory::create($samples)->transpose();

        $this->output()->forward($input);

        return $this;
    }

    /**
     * Backpropagate the error determined by the given outcome and return the
     * gradients at each layer.
     *
     * @param  array  $labels
     * @return self
     */
    public function backpropagate(array $labels) : self
    {
        $this->output()->back($labels);

        return $this;
    }

    /**
     * Read the weight paramters of the network and return an array of weight
     * matrices.
     *
     * @return array
     */
    public function read() : array
    {
        $layers = [];

        foreach ($this->parametric() as $i => $layer) {
            $layers[$i] = $layer->weights();
        }

        return $layers;
    }

    /**
     * Restore the network parameters from an array of weights indexed by layer
     * then neuron then finally synapse weight.
     *
     * @param  array  $parameters
     * @return void
     */
    public function restore(array $parameters) : void
    {
        foreach ($this->parametric() as $i => $layer) {
            $layer->restore($parameters[$i]);
        }
    }

    /**
     * @return int
     */
    public function count() : int
    {
        return $this->depth();
    }

    /**
     * @param  mixed  $index
     * @param  array  $values
     * @throws \RuntimeException
     * @return void
     */
    public function offsetSet($index, $values) : void
    {
        throw new RuntimeException('Network layers cannot be mutated directly.');
    }

    /**
     * Does a given layer exist in the network.
     *
     * @param  mixed  $index
     * @return bool
     */
    public function offsetExists($index) : bool
    {
        return isset($this->layers[$index]);
    }

    /**
     * @param  mixed  $index
     * @throws \RuntimeException
     * @return void
     */
    public function offsetUnset($index) : void
    {
        throw new RuntimeException('Network layers cannot be mutated directly.');
    }

    /**
     * Return a layer from the network given by index.
     *
     * @param  mixed  $index
     * @throws \InvalidArgumentException
     * @return array
     */
    public function offsetGet($index) : array
    {
        if (!isset($this->layers[$index])) {
            throw new InvalidArgumentException('Network layer does not exist.');
        }

        return $this->layers[$index];
    }
}
