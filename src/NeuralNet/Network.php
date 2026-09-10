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
     * The parameters of the network.
     *
     * @return Traversable<Parameter>
     */
    public function parameters() : Traversable;

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
}
