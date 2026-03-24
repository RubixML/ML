<?php

namespace Rubix\ML\NeuralNet\Networks\Base\Contracts;

use NDArray;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\Layers\Base\Contracts\Hidden;
use Rubix\ML\NeuralNet\Layers\Base\Contracts\Input;
use Rubix\ML\NeuralNet\Layers\Base\Contracts\Output;
use Rubix\ML\NeuralNet\Layers\Base\Contracts\Layer;
use Traversable;

/**
 * Network
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
interface Network
{
    /**
     * Return the layers of the network.
     *
     * @return Traversable<Layer>
     */
    public function layers() : Traversable;

    /**
     * Return the number of trainable parameters in the network.
     *
     * @return int
     */
    public function numParams() : int;

    /**
     * Initialize the parameters of the layers and warm the optimizer cache.
     */
    public function initialize() : void;

    /**
     * Run an inference pass and return the activations at the output layer.
     *
     * @param Dataset $dataset
     * @return NDArray
     */
    public function infer(Dataset $dataset) : NDArray;

    /**
     * @param Labeled $dataset
     * @return float
     */
    public function roundtrip(Labeled $dataset) : float;

    /**
     * Return the input layer.
     *
     * @return Input
     */
    public function input() : Input;

    /**
     * Return an array of hidden layers indexed left to right.
     *
     * @return list<Hidden>
     */
    public function hidden() : array;

    /**
     * Return the output layer.
     *
     * @return Output
     */
    public function output() : Output;
}
