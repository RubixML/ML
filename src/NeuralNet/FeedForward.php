<?php

namespace Rubix\ML\NeuralNet;

use NDArray;
use NumPower;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Encoding;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\Parametric;
use Rubix\ML\NeuralNet\Optimizers\Adaptive;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Traversable;
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
 * @author      Samuel Akopyan <leumas.a@gmail.com>
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
     * @var list<Hidden>
     */
    protected array $hidden = [
        //
    ];

    /**
     * The pathing of the backward pass through the hidden layers.
     *
     * @var list<Hidden>
     */
    protected array $backPass = [
        //
    ];

    /**
     * The output layer of the network.
     *
     * @var Output
     */
    protected Output $output;

    /**
     * The gradient descent optimizer used to train the network.
     *
     * @var Optimizer
     */
    protected Optimizer $optimizer;

    /**
     * Whether to pack the samples.
     *
     * @var bool
     */
    private bool $packSamples;

    /**
     * @param Input $input
     * @param Hidden[] $hidden
     * @param Output $output
     * @param Optimizer $optimizer
     * @param bool $packSamples
     */
    public function __construct(Input $input, array $hidden, Output $output, Optimizer $optimizer, bool $packSamples = false)
    {
        SpecificationChain::with([
            new ExtensionIsLoaded('RubixNumPower'),
            new ExtensionMinimumVersion('RubixNumPower', '0.7.0'),
        ])->check();

        $hidden = array_values($hidden);

        $backPass = array_reverse($hidden);

        $this->input = $input;
        $this->hidden = $hidden;
        $this->output = $output;
        $this->optimizer = $optimizer;
        $this->backPass = $backPass;
        $this->packSamples = $packSamples;
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
     * @return list<Hidden>
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
     * @return Traversable<Layer>
     */
    public function layers() : Traversable
    {
        yield $this->input;

        yield from $this->hidden;

        yield $this->output;
    }

    /**
     * Return the number of trainable parameters in the network.
     *
     * @return int
     */
    public function numParams() : int
    {
        $numParams = 0;

        foreach ($this->layers() as $layer) {
            if ($layer instanceof Parametric) {
                foreach ($layer->parameters() as $parameter) {
                    $numParams += $parameter->param()->size();
                }
            }
        }

        return $numParams;
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

        if ($this->optimizer instanceof Adaptive) {
            foreach ($this->layers() as $layer) {
                if ($layer instanceof Parametric) {
                    foreach ($layer->parameters() as $param) {
                        $this->optimizer->warm($param);
                    }
                }
            }
        }
    }

    /**
     * Run an inference pass and return the activations at the output layer.
     *
     * @param Dataset $dataset
     * @return NDArray
     */
    public function infer(Dataset $dataset) : NDArray
    {
        if ($dataset->empty()) {
            return NumPower::array([]);
        }

        $samples = $this->prepareSamples($dataset);

        $input = NumPower::transpose(NumPower::array($samples), [1, 0]);

        foreach ($this->layers() as $layer) {
            $input = $layer->infer($input);
        }

        return NumPower::transpose($input, [1, 0]);
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
        $input = NumPower::transpose(NumPower::array($dataset->samples()), [1, 0]);

        $this->feed($input);

        $loss = $this->backpropagate($dataset->labels());

        return $loss;
    }

    /**
     * Feed a batch through the network and return a matrix of activations at the output later.
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function feed(NDArray $input) : NDArray
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
        [$gradient, $loss] = $this->output->back($labels, $this->optimizer);

        foreach ($this->backPass as $layer) {
            $gradient = $layer->back($gradient, $this->optimizer);
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

        $layerNum = 0;

        foreach ($this->layers() as $layer) {
            ++$layerNum;

            $dot .= "  N$layerNum [label=\"$layer\",style=\"rounded\"]" . PHP_EOL;

            if ($layerNum > 1) {
                $parentId = $layerNum - 1;

                $dot .= "  N{$parentId} -> N{$layerNum};" . PHP_EOL;
            }
        }

        $dot .= '}';

        return new Encoding($dot);
    }

    /**
     * Prepare samples depending on packing configuration.
     * @param Dataset $dataset
     * @return array
     */
    private function prepareSamples(Dataset $dataset) : array
    {
        $samples = $dataset->samples();

        if (!$this->packSamples) {
            return $samples;
        }

        // Reindex a nested array to ensure all levels have sequential numeric keys
        return array_map('array_values', array_values($samples));
    }
}
