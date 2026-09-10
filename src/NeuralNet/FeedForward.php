<?php

namespace Rubix\ML\NeuralNet;

use Tensor\Matrix;
use Rubix\ML\Encoding;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Traversable;

use function Rubix\ML\enumerate;
use function array_reverse;

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
     * @var Input
     */
    protected Input $input;

    /**
     * The hidden layers of the network.
     *
     * @var list<Layers\Hidden>
     */
    protected array $hidden = [
        //
    ];

    /**
     * The pathing of the backward pass through the hidden layers.
     *
     * @var list<Layers\Hidden>
     */
    protected array $backPass = [
        //
    ];

    /**
     * The output layer.
     *
     * @var Output
     */
    protected Output $output;

    /**
     * @param Input $input
     * @param Layers\Hidden[] $hidden
     * @param Output $output
     * @throws InvalidArgumentException
     */
    public function __construct(Input $input, array $hidden, Output $output)
    {
        $hidden = array_values($hidden);

        $backPass = array_reverse($hidden);

        $this->input = $input;
        $this->hidden = $hidden;
        $this->output = $output;
        $this->backPass = $backPass;
    }

    /**
     * Return the input layer.
     *
     * @return Input
     */
    public function input() : Input
    {
        return $this->input;
    }

    /**
     * Return an array of hidden layers indexed left to right.
     *
     * @return list<Layers\Hidden>
     */
    public function hidden() : array
    {
        return $this->hidden;
    }

    /**
     * Return the output layer.
     *
     * @return Output
     */
    public function output() : Output
    {
        return $this->output;
    }

    /**
     * Return all the layers in the network.
     *
     * @return Traversable<Layers\Layer>
     */
    public function layers() : Traversable
    {
        yield $this->input;

        yield from $this->hidden;

        yield $this->output;
    }

    /**
     * Return the total number of parameters in the network.
     *
     * @return int
     */
    public function numParams() : int
    {
        $numParams = 0;

        foreach ($this->parameters() as $parameter) {
            $numParams += $parameter->param()->size();
        }

        return $numParams;
    }

    /**
     * Return an iterable of all the trainable parameters in the network.
     *
     * @return Traversable<Parameter>
     */
    public function parameters() : Traversable
    {
        foreach ($this->layers() as $layer) {
            if ($layer instanceof Parametric) {
                foreach ($layer->parameters() as $param) {
                    yield $param;
                }
            }
        }
    }

    /**
     * The number of trainable parameters in the network.
     */
    public function numTrainableParams() : int
    {
        $numParams = 0;

        foreach ($this->trainableParameters() as $parameter) {
            $numParams += $parameter->param()->size();
        }

        return $numParams;
    }

    /**
     * Return an iterable of all the trainable (unfrozen) parameters in the network.
     *
     * @return Traversable<Parameter>
     */
    public function trainableParameters() : Traversable
    {
        foreach ($this->parameters() as $param) {
            if (!$param->frozen()) {
                yield $param;
            }
        }
    }

    /**
     * Initialize the parameters of the layers and warm the optimizer cache.
     */
    public function initialize() : void
    {
        $fanIn = 1;

        foreach ($this->layers() as $layer) {
            $fanIn = $layer->initialize($fanIn);
        }
    }

    /**
     * Freeze the first k hidden layers of the network preventing their
     * parameters from being updated during training.
     *
     * @param int $k
     * @throws RuntimeException
     * @throws InvalidArgumentException
     */
    public function freezeFirstKLayers(int $k) : void
    {
        $numHiddenLayers = count($this->hidden());

        if ($k < 1 or $k > $numHiddenLayers) {
            throw new InvalidArgumentException('Number of layers to freeze'
                . " must be between 1 and $numHiddenLayers, $k given.");
        }

        $firstKLayers = array_slice($this->hidden(), 0, $k);

        foreach ($firstKLayers as $layer) {
            if ($layer instanceof Parametric) {
                foreach ($layer->parameters() as $parameter) {
                    $parameter->freeze();
                }
            }
        }
    }

    /**
     * Unfreeze the hidden layers of the network allowing their parameters to
     * be updated during training.
     *
     * @throws RuntimeException
     */
    public function unfreeze() : void
    {
        foreach ($this->parameters() as $param) {
            $param->unfreeze();
        }
    }

    /**
     * Run an inference pass and return the activations at the output layer.
     *
     * @param Dataset $dataset
     * @return Matrix
     */
    public function infer(Dataset $dataset) : Matrix
    {
        $input = Matrix::quick($dataset->samples())->transpose();

        foreach ($this->layers() as $layer) {
            $input = $layer->infer($input);
        }

        return $input->transpose();
    }

    /**
     * Perform a forward and backward pass of the network in one call. Returns
     * the loss from the backward pass.
     *
     * @param Labeled $dataset
     * @return float
     */
    public function roundtrip(Labeled $dataset) : float
    {
        $input = Matrix::quick($dataset->samples())->transpose();

        $this->feed($input);

        $loss = $this->backpropagate($dataset->labels());

        return $loss;
    }

    /**
     * Feed a batch through the network and return a matrix of activations at the output later.
     *
     * @param Matrix $input
     * @return Matrix
     */
    public function feed(Matrix $input) : Matrix
    {
        foreach ($this->layers() as $layer) {
            $input = $layer->forward($input);
        }

        return $input;
    }

    /**
     * Backpropagate the gradient of the cost function and return the loss.
     *
     * @param list<string|int|float> $labels
     * @return float
     */
    public function backpropagate(array $labels) : float
    {
        [$gradient, $loss] = $this->output->back($labels);

        foreach ($this->backPass as $layer) {
            $gradient = $layer->back($gradient);
        }

        return $loss;
    }

    /**
     * Export the network architecture as a graph in dot format.
     *
     * @return Encoding
     */
    public function exportGraphviz() : Encoding
    {
        $dot = 'digraph Tree {' . PHP_EOL;
        $dot .= '  node [shape=box, fontname=helvetica];' . PHP_EOL;

        foreach (enumerate($this->layers(), 1) as $layerNum => $layer) {
            $dot .= "  N$layerNum [label=\"$layer\",style=\"rounded\"]" . PHP_EOL;

            if ($layerNum > 1) {
                $parentId = $layerNum - 1;

                $dot .= "  N{$parentId} -> N{$layerNum};" . PHP_EOL;
            }
        }

        $dot .= '}';

        return new Encoding($dot);
    }
}
