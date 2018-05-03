<?php

namespace Rubix\Engine\NeuralNet\Layers;

use Rubix\Engine\NeuralNet\ActivationFunctions\Identity;

class Maxout extends Hidden
{
    /**
     * @param  array  $n
     * @return void
     */
    public function __construct(int $n)
    {
        parent::__construct($n, new Identity());
    }

    /**
     * Return a softmax of the activations.
     *
     * @return array
     */
    public function fire() : array
    {
        $activations = parent::fire();

        $max = max($activations);

        return array_replace(array_fill(0, $n, 0.0), [array_search($max, $activations) => $max]);
    }
}
