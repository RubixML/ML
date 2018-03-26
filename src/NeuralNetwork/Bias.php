<?php

namespace Rubix\Engine\NeuralNetwork;

class Bias extends Neuron
{
    /**
     * @return float
     */
    public function output() : float
    {
        return 1.0;
    }
}
