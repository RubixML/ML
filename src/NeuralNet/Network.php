<?php

namespace Rubix\ML\NeuralNet;

use Generator;

interface Network
{
    /**
     * Return the layers of the network.
     *
     * @return \Generator<\Rubix\ML\NeuralNet\Layers\Layer>
     */
    public function layers() : Generator;
}
