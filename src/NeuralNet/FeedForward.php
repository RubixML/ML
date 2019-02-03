<?php

namespace Rubix\ML\NeuralNet;

use Rubix\Tensor\Matrix;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Adaptive;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use InvalidArgumentException;

/**
 * Feed Forward
 *
 * A feed forward neural network implementation consisting of an input and
 * output layer and any number of intermediate hidden layers.
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

        $fanIn = 0;

        foreach ($this->layers as $layer) {
            $fanIn = $layer->initialize($fanIn);
        }

        if ($optimizer instanceof Adaptive) {
            foreach ($this->layers as $layer) {
                if ($layer instanceof Parametric) {
                    foreach ($layer->parameters() as $param) {
                        $optimizer->initialize($param);
                    }
                }
            }
        }

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
     * Do a forward and backward pass of the network in one call. Returns the
     * loss.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $batch
     * @return float
     */
    public function roundtrip(Labeled $batch) : float
    {
        $samples = Matrix::quick($batch->samples())->transpose();

        $this->feed($samples);

        return $this->backpropagate($batch->labels());
    }

    /**
     * Feed a batch through the network and return a matrix of activations.
     *
     * @param  \Rubix\Tensor\Matrix  $input
     * @return \Rubix\Tensor\Matrix
     */
    public function feed(Matrix $input) : Matrix
    {
        foreach ($this->layers as $layer) {
            $input = $layer->forward($input);
        }

        return $input;
    }

    /**
     * Run an inference pass and return the activations at the output layer.
     *
     * @param  \Rubix\Tensor\Matrix  $input
     * @return \Rubix\Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        foreach ($this->layers as $layer) {
            $input = $layer->infer($input);
        }

        return $input;
    }

    /**
     * Backpropagate the gradient produced by the cost function and return the
     * loss.
     *
     * @param  array  $labels
     * @return float
     */
    public function backpropagate(array $labels) : float
    {
        [$gradient, $loss] = $this->output()
            ->back($labels, $this->optimizer);

        foreach ($this->backPass as $layer) {
            $gradient = $layer->back($gradient, $this->optimizer);
        }

        return $loss;
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
