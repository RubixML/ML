<?php

namespace Rubix\ML\NeuralNet\Networks\Base\Contracts;

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
     * @return Traversable
     */
    public function layers() : Traversable;
}
