<?php

namespace Rubix\Engine\NeuralNet\Layers;

use Rubix\Engine\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;

class Multiclass extends Output
{
    /**
     * The labels of the class outcomes.
     *
     * @var array
     */
    protected $labels = [
        //
    ];

    /**
     * @param  array  $labels
     * @param  \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction  $activationFunction
     * @return void
     */
    public function __construct(array $labels, ActivationFunction $activationFunction = null)
    {
        $labels = array_values(array_unique($labels));
        $n = count($labels);

        if (!isset($activationFunction)) {
            $activationFunction = new Sigmoid();
        }

        parent::__construct($n, $activationFunction);

        $this->labels = $labels;
    }

    /**
     * @param  int  $index
     * @return array
     */
    public function labels() : array
    {
        return $this->labels;
    }

    /**
     * Return the label of an output neuron at given offset.
     *
     * @param  int  $index
     * @return string|null
     */
    public function label(int $index) : ?string
    {
        return $this->labels[$index] ?? null;
    }

    /**
     * Return the activations with class labels as keys.
     *
     * @return array
     */
    public function output() : array
    {
        return array_combine($this->labels, parent::output());
    }
}
