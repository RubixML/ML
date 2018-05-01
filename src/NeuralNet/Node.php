<?php

namespace Rubix\Engine\NeuralNet;

interface Node
{
    /**
     * The output of the neuron.
     *
     * @return float
     */
    public function output() : float;
}
