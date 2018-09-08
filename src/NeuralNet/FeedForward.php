<?php

namespace Rubix\ML\NeuralNet;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
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
     * The function that computes the loss of bad activations.
     *
     * @var \Rubix\ML\NeuralNet\CostFunctions\CostFunction
     */
    protected $costFunction;

    /**
     * The gradient descent optimizer used to train the network.
     *
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * The memoized pathing of the backward pass through the hidden layers.
     *
     * @var array
     */
    protected $backPass = [
        //
    ];

    /**
     * @param  \Rubix\ML\NeuralNet\Layers\Input  $input
     * @param  array  $hidden
     * @param  \Rubix\ML\NeuralNet\Layers\Output  $output
     * @param  \Rubix\ML\NeuralNet\CostFunctions\CostFunction  $costFunction
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @return void
     */
    public function __construct(Input $input, array $hidden, Output $output,
                                CostFunction $costFunction, Optimizer $optimizer)
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

        $this->initialize();

        $this->costFunction = $costFunction;
        $this->optimizer = $optimizer;
        $this->backPass = array_reverse($this->hidden());
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
        return array_slice($this->layers, 1, $this->depth() - 2, true);
    }

    /**
     * Return the output layer.
     *
     * @return \Rubix\ML\NeuralNet\Layers\Output
     */
    public function output() : Output
    {
        return $this->layers[$this->depth() - 1];
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
     * Initialize the layers in the network.
     *
     * @return void
     */
    public function initialize() : void
    {
        $fanIn = 0;

        foreach ($this->layers as $layer) {
            $fanIn = $layer->init($fanIn);
        }
    }

    /**
     * Do a forward and backward pass of the network in one call. Returns the
     * loss.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $batch
     * @return float
     */
    public function roundtrip(Labeled $batch) : float
    {
        return $this->feed($batch)->backpropagate($batch);
    }

    /**
     * Feed a sample through the network and return the output activations
     * of each output neuron.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $batch
     * @return self
     */
    public function feed(Labeled $batch) : self
    {
        $input = Matrix::build($batch->samples())
            ->transpose();

        foreach ($this->layers as $layer) {
            $input = $layer->forward($input);
        }

        return $this;
    }

    /**
     * Backpropagate the gradients from the previous layer and take a step
     * in the direction of the steepest descent.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $batch
     * @return float
     */
    public function backpropagate(Labeled $batch) : float
    {
        list($prevGradients, $cost) = $this->output()
            ->back($batch->labels(), $this->costFunction, $this->optimizer);

        foreach ($this->backPass as $layer) {
            $prevGradients = $layer->back($prevGradients, $this->optimizer);
        }

        return $cost;
    }

    /**
     * Return the activations of the neurons at the output layer.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function infer(Dataset $dataset) : array
    {
        $input = Matrix::build($dataset->samples())
            ->transpose();

        foreach ($this->layers as $layer) {
            $input = $layer->infer($input);
        }

        return $input->transpose()->asArray();
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
