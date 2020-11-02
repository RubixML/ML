<?php

namespace Rubix\ML\NeuralNet;

use Tensor\Matrix;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Adaptive;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;
use Traversable;

/**
 * Feed Forward
 *
 * A feed forward neural network implementation consisting of an input and
 * output layer and any number of intermediate hidden layers.
 *
 * @internal
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
     * The pathing of the backward pass through the hidden layers.
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
                throw new InvalidArgumentException('Hidden layer must'
                    . ' implement the Hidden interface.');
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
        $this->output = $output;
        $this->backPass = array_reverse($hidden);
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
     * @return \Traversable<\Rubix\ML\NeuralNet\Layers\Layer>
     */
    public function layers() : Traversable
    {
        yield $this->input;

        foreach ($this->hidden as $hidden) {
            yield $hidden;
        }

        yield $this->output;
    }

    /**
     * Run an inference pass and return the activations at the output layer.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return \Tensor\Matrix
     */
    public function infer(Dataset $dataset) : Matrix
    {
        $input = Matrix::quick($dataset->samples())->transpose();

        $input = $this->input->infer($input);

        foreach ($this->hidden as $hidden) {
            $input = $hidden->infer($input);
        }

        return $this->output->infer($input)->transpose();
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
        $input = Matrix::quick($dataset->samples())->transpose();

        $this->feed($input);

        return $this->backpropagate($dataset->labels());
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

        return $this->output->forward($input);
    }

    /**
     * Backpropagate the gradient produced by the cost function and return the loss.
     *
     * @param list<string|int|float> $labels
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
}
