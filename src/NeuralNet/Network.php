<?php

namespace Rubix\ML\NeuralNet;

use Generator;

interface Network
{
    /**
     * Return all the layers in the network.
     *
     * @return \Generator
     */
    public function layers() : Generator;

    /**
     * Return the parametric layers of the network.
     *
     * @return \Generator
     */
    public function parametric() : Generator;

    /**
     * The depth of the network. i.e. the number of parametric layers.
     *
     * @return int
     */
    public function depth() : int;
}
