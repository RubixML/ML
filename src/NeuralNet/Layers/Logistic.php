<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use InvalidArgumentException;

class Logistic implements Output
{
    /**
     * The labels of either of the possible outcomes.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * The function that outputs the activation or implulse of each neuron.
     *
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction
     */
    protected $activationFunction;

    /**
     * The L2 regularization parameter.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The weight matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $weights;

    /**
     * The memoized input matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $input;

    /**
     * The memoized z matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $z;

    /**
     * The memoized output activations matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $computed;

    /**
     * The memoized gradient matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $gradients;

    /**
     * @param  array  $labels
     * @param  float  $alpha
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $labels, float $alpha = 1e-4)
    {
        $labels = array_unique($labels);

        if (count($labels) !== 2) {
            throw new InvalidArgumentException('The number of unique class'
                . ' labels must be exactly 2.');
        }

        $this->classes = [$labels[0] => 0, $labels[1] => 1];
        $this->activationFunction = new Sigmoid();
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
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function weights() : Matrix
    {
        return $this->weights;
    }

    /**
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function gradients() : Matrix
    {
        return $this->gradients;
    }

    /**
     * Initialize the layer by fully connecting each neuron to every input and
     * generating a random weight for each parameter/synapse in the layer.
     *
     * @param  int  $width
     * @return void
     */
    public function initialize(int $width) : void
    {
        $weights = array_fill(0, $this->width(),
            array_fill(0, $width, 0.0));

        $r = sqrt(6 / $width);

        for ($i = 0; $i < $this->width(); $i++) {
            for ($j = 0; $j < $width; $j++) {
                $weights[$i][$j] = random_int((int) (-$r * 1e8),
                    (int) ($r * 1e8)) / 1e8;
            }
        }

        $this->weights = new Matrix($weights);
    }

    /**
     * Compute the input sum and activation of each neuron in the layer and return
     * an activation matrix.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $input
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $this->input = $input;

        $this->z = $this->weights->multiply($input);

        $this->computed = $this->activationFunction->compute($this->z);

        return $this->computed;
    }

    /**
     * Calculate the errors and gradients for each output neuron.
     *
     * @param  array  $labels
     * @return array
     */
    public function back(array $labels) : array
    {
        $errors = [[]];

        foreach ($labels as $i => $label) {
            $expected = $this->classes[$label];

            if ($expected === 0) {
                $error = (1 - $expected) * log(1 - $this->computed[0][$i]);
            } else {
                $error = -$expected * log($this->computed[0][$i]);
            }

            $errors[0][$i] = $error + 0.5 * $this->alpha
                * array_sum($this->weights[0]) ** 2;
        }

        $errors = new Matrix($errors);

        $errors = $this->activationFunction
            ->differentiate($this->z, $this->computed)
            ->hadamardProduct($errors);

        $this->gradients = $errors->multiply($this->input->transpose());

        return [$this->weights, $errors];
    }

    /**
     * Return the computed activation matrix.
     *
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function activations() : Matrix
    {
        return $this->computed;
    }

    /**
     * Update the parameters in the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $steps
     * @return void
     */
    public function update(Matrix $steps) : void
    {
        $this->weights = $this->weights->add($steps);
    }

    /**
     * Restore the parameters in the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $weights
     * @return void
     */
    public function restore(Matrix $weights) : void
    {
        $this->weights = $weights;
    }
}
