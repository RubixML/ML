<?php

namespace Rubix\ML\NeuralNet;

use Traversable;

/**
 * Network
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Network
{
    /**
     * Return the total number of parameters in the network.
     *
     * @return int
     */
    public function numParams() : int;

    /**
     * The parameters of the network.
     *
     * @return Traversable<Parameter>
     */
    public function parameters() : Traversable;

    /**
     * The number of trainable parameters in the network.
     *
     * @return int
     */
    public function numTrainableParams() : int;

    /**
     * Return an iterable of all the trainable (unfrozen) parameters in the network.
     *
     * @return Traversable<Parameter>
     */
    public function trainableParameters() : Traversable;

    /**
     * Return the layers of the network.
     *
     * @return Traversable<Layers\Layer>
     */
    public function layers() : Traversable;

    /**
     * Initialize the parameters of the layers.
     */
    public function initialize() : void;
}
