<?php

namespace Rubix\ML\NeuralNet;

use Traversable;

interface Network
{
    /**
     * Return the layers of the network.
     *
     * @return \Traversable<\Rubix\ML\NeuralNet\Layers\Layer>
     */
    public function layers() : Traversable;
}
