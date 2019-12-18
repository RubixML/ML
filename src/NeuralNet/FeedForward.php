<?php

namespace Rubix\ML\NeuralNet;

use Tensor\Tensor;
use Tensor\Matrix;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Adaptive;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;
use Generator;

use function count;
use function get_class;

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
     * The input layer to the network.
     *
     * @var \Rubix\ML\NeuralNet\Layers\Input
     */
    protected $input;

    /**
     * The hidden layers of the network.
     *
     * @var \Rubix\ML\NeuralNet\Layers\Hidden[]
     */
    protected $hidden = [
        //
    ];

    /**
     * The memoized pathing of the backward pass through the hidden layers.
     *
     * @var \Rubix\ML\NeuralNet\Layers\Hidden[]
     */
    protected $backPass = [
        //
    ];

    /**
     * The output layer of the network.
     *
     * @var \Rubix\ML\NeuralNet\Layers\Output
     */
    protected $output;

    /**
     * The gradient descent optimizer used to train the network.
     *
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * @param \Rubix\ML\NeuralNet\Layers\Input $input
     * @param \Rubix\ML\NeuralNet\Layers\Hidden[] $hidden
     * @param \Rubix\ML\NeuralNet\Layers\Output $output
     * @param \Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     */
    public function __construct(Input $input, array $hidden, Output $output, Optimizer $optimizer)
    {
        foreach ($hidden as $layer) {
            if (!$layer instanceof Hidden) {
                throw new InvalidArgumentException('Cannot use '
                    . get_class($layer) . ' as a hidden layer.');
            }
        }

        $layers = [$input];
        $layers = array_merge($layers, $hidden);
        $layers[] = $output;

        foreach ($layers as $layer) {
            $fanIn = $layer->initialize($fanIn ?? 0);
        }

        if ($optimizer instanceof Adaptive) {
            foreach ($layers as $layer) {
                if ($layer instanceof Parametric) {
                    foreach ($layer->parameters() as $param) {
                        $optimizer->warm($param);
                    }
                }
            }
        }

        $this->input = $input;
        $this->hidden = $hidden;
        $this->backPass = array_reverse($hidden);
        $this->output = $output;
        $this->optimizer = $optimizer;
    }

    /**
     * Return the input layer.
     *
     * @return \Rubix\ML\NeuralNet\Layers\Input
     */
    public function input() : Input
    {
        return $this->input;
    }

    /**
     * Return an array of hidden layers indexed left to right.
     *
     * @return \Rubix\ML\NeuralNet\Layers\Hidden[]
     */
    public function hidden() : array
    {
        return $this->hidden;
    }

    /**
     * Return the output layer.
     *
     * @return \Rubix\ML\NeuralNet\Layers\Output
     */
    public function output() : Output
    {
        return $this->output;
    }

    /**
     * Return all the layers in the network.
     *
     * @return \Generator<\Rubix\ML\NeuralNet\Layers\Layer>
     */
    public function layers() : Generator
    {
        yield $this->input;
        
        foreach ($this->hidden as $hidden) {
            yield $hidden;
        }

        yield $this->output;
    }

    /**
     * The parametric layers of the network. i.e. the layers that have weights.
     *
     * @return \Generator<\Rubix\ML\NeuralNet\Layers\Parametric>
     */
    public function parametric() : Generator
    {
        foreach ($this->hidden as $layer) {
            if ($layer instanceof Parametric) {
                yield $layer;
            }
        }

        if ($this->output instanceof Parametric) {
            yield $this->output;
        }
    }

    /**
     * The depth of the network. i.e. the number of layers including
     * input, hidden, and output.
     *
     * @return int
     */
    public function depth() : int
    {
        return count($this->hidden) + 2;
    }

    /**
     * Perform a forward and backward pass of the network in one call. Returns
     * the loss from the backward pass.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return float
     */
    public function roundtrip(Labeled $dataset) : float
    {
        $this->feed(Matrix::quick($dataset->samples())->transpose());
        
        $loss = $this->backpropagate($dataset->labels());

        return $loss;
    }

    /**
     * Feed a batch through the network and return a matrix of activations.
     *
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function feed(Matrix $input) : Matrix
    {
        $input = $this->input->forward($input);

        foreach ($this->hidden as $hidden) {
            $input = $hidden->forward($input);
        }

        $activations = $this->output->forward($input);

        return $activations;
    }

    /**
     * Run an inference pass and return the activations at the output layer.
     *
     * @param \Tensor\Matrix $input
     * @return \Tensor\Matrix
     */
    public function infer(Matrix $input) : Tensor
    {
        $input = $this->input->infer($input);

        foreach ($this->hidden as $hidden) {
            $input = $hidden->infer($input);
        }

        $activations = $this->output->infer($input);

        return $activations;
    }

    /**
     * Backpropagate the gradient produced by the cost function and return
     * the loss calculated by the output layer's cost function.
     *
     * @param (string|int|float)[] $labels
     * @return float
     */
    public function backpropagate(array $labels) : float
    {
        [$gradient, $loss] = $this->output->back($labels, $this->optimizer);

        foreach ($this->backPass as $layer) {
            $gradient = $layer->back($gradient, $this->optimizer);
        }

        return $loss;
    }

    /**
     * Restore the network parameters from a snapshot.
     *
     * @param \Rubix\ML\NeuralNet\Snapshot<array> $snapshot
     */
    public function restore(Snapshot $snapshot) : void
    {
        foreach ($snapshot as [$layer, $params]) {
            if ($layer instanceof Parametric) {
                $layer->restore($params);
            }
        }
    }
}
