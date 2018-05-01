<?php

namespace Rubix\Engine\NeuralNet;

class Bias implements Node
{
    /**
     * @return float
     */
    public function output() : float
    {
        return 1.0;
    }
}
