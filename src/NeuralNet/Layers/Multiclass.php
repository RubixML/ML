<?php

namespace Rubix\Engine\NeuralNet\Layers;

use Rubix\Engine\NeuralNet\ActivationFunctions\Sigmoid;
use InvalidArgumentException;

class Multiclass implements Output, Parametric
{
    /**
     * The labels of each possible outcome.
     *
     * @var array
     */
    protected $labels = [
        //
    ];

    /**
     * The function that outputs the activation or implulse of each neuron.
     *
     * @var \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction
     */
    protected $activationFunction;

    /**
     * The L2 regularization term.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The array of weight vectors for each synapse in the layer.
     *
     * @var array
     */
    protected $weights = [
        //
    ];

    /**
     * The memoized inputs to the layer.
     *
     * @var array
     */
    protected $inputs = [
        //
    ];

    /**
     * The memoized z values for each neuron in the output layer.
     *
     * @var array
     */
    protected $sigmas = [
        //
    ];

    /**
     * The memoized actvations for each neuron in the output layer.
     *
     * @var array
     */
    protected $computed = [
        //
    ];

    /**
     * @param  array  $labels
     * @param  float  $alpha
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $labels, float $alpha = 1e-4)
    {
        $labels = array_values(array_unique($labels));

        if (count($labels) < 1) {
            throw new InvalidArgumentException('The number of unique labeled outcomes must be greater than 0.');
        }

        $this->labels = $labels;
        $this->activationFunction = new Sigmoid();
        $this->alpha = $alpha;
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return count($this->labels);
    }

    /**
     * Return the in degree for each neuron in the layer.
     *
     * @return array
     */
    public function inDegrees() : array
    {
        return array_map(function ($neuron) {
            return count($neuron);
        }, $this->weights);
    }

    /**
     * @return array
     */
    public function parameters() : array
    {
        return $this->weights;
    }

    /**
     * Initialize the layer by fully connecting each neuron to every input and
     * generating a random weight for each parameter/synapse in the layer.
     *
     * @param  \Rubix\Engine\NeuralNet\Layers\Layer  $previous
     * @return void
     */
    public function initialize(Layer $previous) : void
    {
        for ($i = 0; $i < $this->width(); $i++) {
            for ($j = 0; $j < $previous->width(); $j++) {
                $this->weights[$i][$j] = $this->activationFunction->initialize($previous->width());
            }
        }
    }

    /**
     * Compute the input sum and activation of each nueron in the layer and return
     * an activation vector.
     *
     * @param  array  $inputs
     * @return array
     */
    public function forward(array $inputs) : array
    {
        $this->inputs = $inputs;
        $activations = [];

        for ($i = 0; $i < $this->width(); $i++) {
            $z = 0.0;

            foreach ($inputs as $j => $input) {
                $z += $this->weights[$i][$j] * $input;
            }

            $this->sigmas[$i] = $z;
            $activations[$i] = $this->activationFunction->compute($z);
        }

        $this->computed = $activations;

        return array_combine($this->labels, $activations);
    }

    /**
     * Calculate a backward pass and return an array of erros and gradients.
     *
     * @param  mixed  $outcome
     * @return array
     */
    public function back($outcome) : array
    {
        $errors = $gradients = [];

         foreach ($this->labels as $i => $label) {
             $expected = $label === $outcome ? 1.0 : 0.0;

             $slope = $this->activationFunction->differentiate($this->sigmas[$i], $this->computed[$i]);

             $errors[$i] = $slope * ($expected - $this->computed[$i]);

             for ($j = 0; $j < count($this->weights[$i]); $j++) {
                 $gradients[$i][$j] = $errors[$i] * $this->inputs[$j];
             }
         }

         return [$errors, $gradients];
    }

    /**
     * Update the parameters in the layer.
     *
     * @param  array  $steps
     * @return void
     */
    public function update(array $steps) : void
    {
        foreach ($this->weights as $i => &$neuron) {
            foreach ($neuron as $j => &$weight) {
                $weight -= $this->alpha * $weight;
                $weight += $steps[$i][$j];
            }
        }
    }

    /**
     * Set the parameters in the layer.
     *
     * @param  array  $weights
     * @return void
     */
    public function setParameters(array $weights) : void
    {
        foreach ($this->weights as $i => &$neuron) {
            foreach ($neuron as $j => &$weight) {
                $weight = $weights[$i][$j];
            }
        }
    }
}
