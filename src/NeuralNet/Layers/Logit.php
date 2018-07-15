<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use InvalidArgumentException;

/**
 * Logit
 *
 * This Logit layer consists of a single Sigmoid neuron capable of
 * distinguishing between two classes. The Logit layer is useful for neural
 * networks that output a binary class prediction.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Logit implements Output
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
     * The width of the layer. i.e. the number of neurons.
     *
     * @var int
     */
    protected $width;

    /**
     * The weight matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix|null
     */
    protected $weights;

    /**
     * The memoized input matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix|null
     */
    protected $input;

    /**
     * The memoized z matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix|null
     */
    protected $z;

    /**
     * The memoized activation matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix|null
     */
    protected $computed;

    /**
     * The memoized gradient matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix|null
     */
    protected $gradients;

    /**
     * The gradient descent optimizer.
     *
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer|null
     */
    protected $optimizer;

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

        if ($alpha < 0) {
            throw new InvalidArgumentException('L2 regularization parameter'
                . ' must be 0 or greater.');
        }

        $this->classes = [$labels[0] => 0, $labels[1] => 1];
        $this->activationFunction = new Sigmoid();
        $this->alpha = $alpha;
        $this->width = 1;
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return $this->width;
    }

    /**
     * Initialize the layer by fully connecting each neuron to every input and
     * generating a random weight for each parameter/synapse in the layer.
     *
     * @param  int  $prevWidth
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @return int
     */
    public function initialize(int $prevWidth, Optimizer $optimizer) : int
    {
        $r = sqrt(6 / $prevWidth);

        $weights = [[]];

        for ($i = 0; $i < $this->width; $i++) {
            for ($j = 0; $j < $prevWidth; $j++) {
                $weights[$i][$j] = rand((int) (-$r * 1e8), (int) ($r * 1e8)) / 1e8;
            }
        }

        $this->weights = new Matrix($weights);
        $this->optimizer = $optimizer;

        $this->optimizer->initialize($this->weights);

        return $this->width;
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
        $l2penalty = 0.5 * $this->alpha * array_sum($this->weights[0]) ** 2;

        $errors = [];

        foreach ($this->computed->getRow(0) as $i => $activation) {
            $expected = $this->classes[$labels[$i]];

            $errors[$i] = ($expected - $activation) + $l2penalty;
        }

        $errors = new Matrix([$errors]);

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
     * Update the parameters in the layer and return the magnitude of the step.
     *
     * @return float
     */
    public function update() : float
    {
        $steps = $this->optimizer->step($this->gradients);

        $this->weights = $this->weights->add($steps);

        return $steps->maxNorm();
    }

    /**
     * @return array
     */
    public function read() : array
    {
        return [
            'weights' => clone $this->weights,
        ];
    }

    /**
     * Restore the parameters of the layer.
     *
     * @param  array  $parameters
     * @return void
     */
    public function restore(array $parameters) : void
    {
        $this->weights = $parameters['weights'];
    }
}
