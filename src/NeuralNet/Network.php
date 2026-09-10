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
     * Return the layers of the network.
     *
     * @return Traversable<Layers\Layer>
     */
    public function layers() : Traversable;
}
