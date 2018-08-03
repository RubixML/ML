<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Parameter;
use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use InvalidArgumentException;
use RuntimeException;

/**
 * Binomial
 *
 * This Binomial layer consists of a single Sigmoid neuron capable of
 * distinguishing between two discrete classes. The Binomial layer is useful for
 * neural networks that output a binary class prediction such as yes or no.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Binomial implements Output
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
     * The function that outputs the activation or implulse of each neuron.
     *
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction
     */
    protected $activationFunction;

    /**
     * The weights.
     *
     * @var \Rubix\ML\NeuralNet\Parameter
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
        $this->alpha = $alpha;
        $this->activationFunction = new Sigmoid();
        $this->width = 1;
        $this->weights = new Parameter(new Matrix([]));
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
     * @param  int  $fanIn
     * @return int
     */
    public function init(int $fanIn) : int
    {
        $r = sqrt(6 / $fanIn);

        $min = (int) (-$r * self::SCALE);
        $max = (int) ($r * self::SCALE);

        $w = [[]];

        for ($i = 0; $i < $this->width; $i++) {
            for ($j = 0; $j < $fanIn; $j++) {
                $w[$i][$j] = rand($min, $max) / self::SCALE;
            }
        }

        $this->weights = new Parameter(new Matrix($w));

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

        $this->z = $this->weights->w()->multiply($input);

        $this->computed = $this->activationFunction->compute($this->z);

        return $this->computed;
    }

    /**
     * Compute the inferential activations of each neuron in the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $input
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        $z = $this->weights->w()->multiply($input);

        return $this->activationFunction->compute($z);
    }

    /**
     * Calculate the errors and gradients for each output neuron and update.
     *
     * @param  array  $labels
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @throws \RuntimeException
     * @return array
     */
    public function back(array $labels, Optimizer $optimizer) : array
    {
        if (is_null($this->input) or is_null($this->z) or is_null($this->computed)) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $penalty = 0.5 * $this->alpha
            * array_sum($this->weights->w()->getRow(0)) ** 2;

        $errors = [];

        foreach ($this->computed->getRow(0) as $i => $activation) {
            $expected = $this->classes[$labels[$i]];

            $errors[$i] = ($expected - $activation) + $penalty;
        }

        $errors = new Matrix([$errors]);

        $errors = $this->activationFunction
            ->differentiate($this->z, $this->computed)
            ->hadamardProduct($errors);

        $gradient = $errors->multiply($this->input->transpose());

        $step = $optimizer->step($this->weights, $gradient);

        $this->weights->update($step);

        unset($this->input, $this->z, $this->computed);

        return [$this->weights->w(), $errors, $step->maxNorm()];
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
