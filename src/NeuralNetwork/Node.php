<?php

namespace Rubix\Engine\NeuralNetwork;

interface Node
{
    /**
     * The output of the node.
     *
     * @return float
     */
    public function output() : float;
}
