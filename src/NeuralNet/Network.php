<?php

namespace Rubix\ML\NeuralNet;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Output;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;
use RuntimeException;

class Network
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
     * The gradient descent optimizer used to train the network.
     *
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * @param  \Rubix\ML\NeuralNet\Layers\Input  $input
     * @param  array  $hidden
     * @param  \Rubix\ML\NeuralNet\Layers\Output  $output
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @return void
     */
    public function __construct(Input $input, array $hidden, Output $output, Optimizer $optimizer)
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

        $this->optimizer = $optimizer;
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
     * @return void
     */
    public function initialize() : void
    {
        for ($i = 1; $i < count($this->layers); $i++) {
            $current = $this->layers[$i];
            $previous = $this->layers[$i - 1];

            $current->initialize($previous);

            $this->optimizer->initialize($current);
        }
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
     * Return the activations of the neurons at the output layer.
     *
     * @return array
     */
    public function activations() : array
    {
        return $this->output()->computed()->transpose()->getMatrix();
    }

    /**
     * Take a step of gradient descent and return the magnitude.
     *
     * @return float
     */
    public function step() : float
    {
        $magnitude = 0.0;

        foreach ($this->parametric() as $layer) {
            $step = $this->optimizer->step($layer);

            $layer->update($step);

            $magnitude += $step->oneNorm();
        }

        return $magnitude;
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
     * Restore the network parameters from an array of weights indexed by layer.
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
}
