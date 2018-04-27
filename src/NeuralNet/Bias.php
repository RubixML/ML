<?php

namespace Rubix\Engine\NeuralNet;

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
