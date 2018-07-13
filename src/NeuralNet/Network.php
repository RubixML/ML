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
     * The memoized pathing of the backward pass.
     *
     * @var array
     */
    protected $backPath = [
        //
    ];

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

        $prevWidth = $this->input()->width();

        foreach ($this->parametric() as $layer) {
            $prevWidth = $layer->initialize($prevWidth, clone $optimizer);
        }

        $this->backPath = array_reverse($this->parametric());
    }

    /**
     * Return all the layers in the network.
     *
     * @return array
     */
    public function layers() : array
    {
        return $this->layers;
    }

    /**
     * Return the input layer.
     *
     * @return \Rubix\ML\NeuralNet\Layers\Input
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
     * @return \Rubix\ML\NeuralNet\Layers\Output
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
        return array_slice($this->layers, 1, count($this->layers), true);
    }

    /**
     * The depth of the network. i.e. the number of parametric layers.
     *
     * @return int
     */
    public function depth() : int
    {
        return count($this->parametric());
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

        foreach ($this->layers as $layer) {
            $input = $layer->forward($input);
        }

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
        $errors = $weights = null;

        foreach ($this->backPath as $layer) {
            if ($layer instanceof Output) {
                list($weights, $errors) = $layer->back($labels);
            } else {
                list($weights, $errors) = $layer->back($weights, $errors);
            }
        }

        return $this;
    }

    /**
     * Return the activations of the neurons at the output layer.
     *
     * @return array
     */
    public function activations() : array
    {
        return $this->output()->activations()->transpose()->getMatrix();
    }

    /**
     * Take a step of gradient descent and return the magnitude.
     *
     * @return float
     */
    public function step() : float
    {
        $magnitude = 0.0;

        foreach ($this->backPath as $layer) {
            $magnitude += $layer->update();
        }

        return $magnitude;
    }

    /**
     * Read the weight paramters of the network and return a snapshot.
     *
     * @return \Rubix\ML\NeuralNet\Snapshot
     */
    public function read() : Snapshot
    {
        return new Snapshot($this->parametric());
    }


    /**
     * Restore the network parameters from a snapshot.
     *
     * @param  \Rubix\ML\NeuralNet\Snapshot  $snapshot
     * @return void
     */
    public function restore(Snapshot $snapshot) : void
    {
        foreach ($snapshot as $layer) {
            if ($layer instanceof Parametric) {
                $layer->restore($snapshot[$layer]);
            }
        }
    }
}
