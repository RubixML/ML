<?php

namespace Rubix\Engine\NeuralNet\Layers;

use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;
use InvalidArgumentException;

class Dense implements Hidden, Parametric
{
    /**
     * The number of neurons in this layer.
     *
     * @var int
     */
    protected $neurons;

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
     * An array of n-d weight vectors, one per neuron in the layer, where n is
     * the number of nerve synapses.
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
     * The memoized weighted sums or z values upon forward pass for each neuron
     * in the layer.
     *
     * @var array
     */
    protected $sigmas = [
        //
    ];

    /**
     * The computed actvations for each neuron upon forward pass in the layer.
     *
     * @var array
     */
    protected $computed = [
        //
    ];

    /**
     * @param  int  $neurons
     * @param  \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction  $activationFunction
     * @param  float  $alpha
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $neurons, ActivationFunction $activationFunction, float $alpha = 1e-4)
    {
        if ($neurons < 1) {
            throw new InvalidArgumentException('The number of neurons must be greater than 0.');
        }

        $this->neurons = $neurons;
        $this->activationFunction = $activationFunction;
        $this->alpha = $alpha;
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return $this->neurons + 1;
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
     * @param  \Rubix\Engine\NeuralNet\Layers\Layer
     * @return void
     */
    public function initialize(Layer $previous) : void
    {
        for ($i = 0; $i < $this->neurons; $i++) {
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

        for ($i = 0; $i < $this->neurons; $i++) {
            $z = 0.0;

            foreach ($inputs as $j => $input) {
                $z += $this->weights[$i][$j] * $input;
            }

            $this->sigmas[$i] = $z;
            $activations[$i] = $this->activationFunction->compute($z);
        }

        $this->computed = $activations;

        $activations[] = 1.0;

        return $activations;
    }

    /**
     * Calculate a backward pass and return an array of erros and gradients.
     *
     * @param  array  $previousWeights
     * @param  array  $previousErrors
     * @return array
     */
    public function back(array $previousWeights, array $previousErrors) : array
    {
        $errors = $gradients = [];

        for ($i = 0; $i < $this->neurons; $i++) {
            $previousError = 0.0;

            foreach ($previousWeights as $j => $neuron) {
                $previousError += $neuron[$i] * $previousErrors[$j];
            }

            $slope = $this->activationFunction->differentiate($this->sigmas[$i], $this->computed[$i]);

            $errors[$i] = $slope * $previousError;

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
