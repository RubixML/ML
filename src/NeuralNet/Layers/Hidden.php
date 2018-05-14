<?php

namespace Rubix\Engine\NeuralNet\Layers;

interface Hidden extends Layer
{
    /**
     * Backpropagate the error from the previous layer.
     *
     * @param  array  $previousWeights
     * @param  array  $previousErrors
     * @return array
     */
    public function back(array $previousWeights, array $previousErrors) : array;
}
