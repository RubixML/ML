<?php

namespace Rubix\Engine\NeuralNet\Layers;

use Rubix\Engine\NeuralNet\ActivationFunctions\HyperbolicTangent;
use InvalidArgumentException;

class Binary implements Output
{
    /**
     * The labels of either of the possible outcomes.
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
     * The L2 regularization parameter.
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
     * The memoized sum of the weighted inputs.
     *
     * @var float
     */
    protected $z;

    /**
     * The memoized actvations pf the output neuron.
     *
     * @var float
     */
    protected $computed;

    /**
     * @param  array  $labels
     * @param  float  $alpha
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $labels, float $alpha = 1e-4)
    {
        $labels = array_values(array_unique($labels));

        if (count($labels) !== 2) {
            throw new InvalidArgumentException('The number of unique class labels must be exactly 2.');
        }

        $this->labels = [1 => $labels[0], -1 => $labels[1]];
        $this->activationFunction = new HyperbolicTangent();
        $this->alpha = $alpha;
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return 1;
    }

    /**
     * Return the in degree for each neuron in the layer.
     *
     * @return array
     */
    public function inDegrees() : array
    {
        return [count($this->weights)];
    }

    /**
     * @return array
     */
    public function parameters() : array
    {
        return [$this->weights];
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
        for ($i = 0; $i < $previous->width(); $i++) {
            $this->weights[$i] = $this->activationFunction
                ->initialize($previous->width());
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
        $activation = 0.0;
        $z = 0.0;

        foreach ($inputs as $i => $input) {
            $z += $this->weights[$i] * $input;
        }

        $activation = $this->activationFunction->compute($z);

        $this->z = $z;
        $this->computed = $activation;

        $outcome = $this->labels[($activation >= 0 ? 1 : -1)];

        return [$outcome => abs($activation)];
    }

    /**
     * Calculate the error for each neuron based on the outcome and return a
     * gradient vector.
     *
     * @param  mixed  $outcome
     * @return array
     */
    public function back($outcome) : array
    {
        $gradient = [];

        $expected = array_search($outcome, $this->labels);

        $slope = $this->activationFunction
            ->differentiate($this->z, $this->computed);

        $error = $slope * ($expected - $this->computed)
            + 0.5 * $this->alpha * array_sum($this->weights) ** 2;

         foreach ($this->weights as $i => $weight) {
             $gradient[$i] = $error * $this->inputs[$i];
         }

         return [[$error], [$gradient]];
    }

    /**
     * Update the parameters in the layer.
     *
     * @param  array  $steps
     * @return void
     */
    public function update(array $steps) : void
    {
        foreach ($this->weights as $i => &$weight) {
            $weight += $steps[0][$i];
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
        foreach ($this->weights as $i => &$weight) {
            $weight = $weights[0][$i];
        }
    }
}
