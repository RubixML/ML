<?php

namespace Rubix\ML\NeuralNet\Layers\Base\Contracts;

use NDArray;
use Stringable;

/**
 * Hidden
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 * @author      Samuel Akopyan <leumas.a@gmail.com>
 */
interface Layer extends Stringable
{
    /**
     * The width of the layer. i.e. the number of neurons or computation nodes.
     *
     * @internal
     *
     * @return positive-int
     */
    public function width() : int;

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @internal
     *
     * @param positive-int $fanIn
     * @return positive-int
     */
    public function initialize(int $fanIn) : int;

    /**
     * Feed the input forward to the next layer in the network.
     *
     * @internal
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function forward(NDArray $input) : NDArray;

    /**
     * Forward pass during inference.
     *
     * @internal
     *
     * @param NDArray $input
     * @return NDArray
     */
    public function infer(NDArray $input) : NDArray;
}
