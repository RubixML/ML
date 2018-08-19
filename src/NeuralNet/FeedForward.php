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

/**
 * Feed Forward
 *
 * A number of the Estimators in Rubix are implemented as a computational graph
 * commonly referred to as a Neural Network due to its inspiration from the
 * human brain. Neural Nets are trained using an iterative process called
 * Gradient Descent and use Backpropagation (sometimes called Reverse Mode
 * Autodiff) to calculate the error of each parameter in the network.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class FeedForward implements Network
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
     * The memoized pathing of the backward pass.
     *
     * @var array
     */
    protected $backPass = [
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

        $fanIn = $this->input()->width();

        foreach ($this->parametric() as $layer) {
            $fanIn = $layer->init($fanIn);
        }

        $this->backPass = array_reverse($this->parametric());
        $this->optimizer = $optimizer;
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
     * Return the activations of the neurons at the output layer.
     *
     * @param  array  $samples
     * @return array
     */
    public function infer(array $samples) : array
    {
        $input = MatrixFactory::create($samples)->transpose();

        foreach ($this->layers as $layer) {
            $input = $layer->infer($input);
        }

        return $input->transpose()->getMatrix();
    }

    /**
     * Backpropagate the error determined by the previous layer and take a step
     * in the direction of the steepest descent.
     *
     * @param  array  $labels
     * @return float
     */
    public function backpropagate(array $labels) : float
    {
        $prevErrors = $prevWeights = null;

        $cost = 0.0;

        foreach ($this->backPass as $layer) {
            if ($layer instanceof Output) {
                list($prevWeights, $prevErrors, $cost)
                    = $layer->back($labels, $this->optimizer);
            } else {
                list($prevWeights, $prevErrors)
                    = $layer->back($prevWeights, $prevErrors, $this->optimizer);
            }
        }

        return $cost;
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
