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
     * Return the layers of the network.
     *
     * @return \Traversable<\Rubix\ML\NeuralNet\Layers\Layer>
     */
    public function layers() : Traversable;
}
