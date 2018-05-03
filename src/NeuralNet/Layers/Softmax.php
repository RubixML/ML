<?php

namespace Rubix\Engine\NeuralNet\Layers;

use Rubix\Engine\NeuralNet\ActivationFunctions\Identity;

class Softmax extends Multiclass
{
    /**
     * @param  array  $labels
     * @return void
     */
    public function __construct(array $labels)
    {
        parent::__construct($labels, new Identity());
    }

    /**
     * Return a softmax of the activations.
     *
     * @return array
     */
    public function output() : array
    {
        $activations = parent::output();

        $total = array_sum($activations);

        foreach ($activations as &$activation) {
            $activation /= $total;
        }

        return $activations;
    }
}
