<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use InvalidArgumentException;
use RuntimeException;

/**
 * Continuous
 *
 * The Continuous output layer consists of a single linear neuron that outputs a
 * scalar value useful for regression problems.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Continuous implements Output
{
    /**
     * The L2 regularization parameter.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The function that computes the cost of an erroneous activation.
     *
     * @var \Rubix\ML\NeuralNet\CostFunctions\CostFunction
     */
    protected $costFunction;

    /**
     * The weights.
     *
     * @var \Rubix\ML\NeuralNet\Parameter
     */
    protected $weights;

    /**
     * The memoized input matrix.
     *
     * @var \Rubix\ML\Other\Structures\Matrix|null
     */
    protected $input;

    /**
     * The memoized output activations matrix.
     *
     * @var \Rubix\ML\Other\Structures\Matrix|null
     */
    protected $computed;

    /**
     * @param  float  $alpha
     * @param  \Rubix\ML\NeuralNet\CostFunctions\CostFunction  $costFunction
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $alpha = 1e-4, CostFunction $costFunction = null)
    {
        if ($alpha < 0.) {
            throw new InvalidArgumentException('L2 regularization parameter'
                . ' must be be non-negative.');
        }

        if (is_null($costFunction)) {
            $costFunction = new LeastSquares();
        }

        $this->alpha = $alpha;
        $this->costFunction = $costFunction;
        $this->weights = new Parameter(new Matrix([[]]));
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return 1;
    }

    /**
     * Initialize the layer by fully connecting each neuron to every input and
     * generating a random weight for each synapse.
     *
     * @param  int  $fanIn
     * @return int
     */
    public function init(int $fanIn) : int
    {
        $r = (6 / $fanIn) ** 0.25;

        $min = (int) round(-$r * self::PHI);
        $max = (int) round($r * self::PHI);

        $w = [[]];

        for ($i = 0; $i < $fanIn; $i++) {
            $w[$i] = rand($min, $max) / self::PHI;
        }

        $this->weights = new Parameter(new Matrix([$w]));

        return $this->width();
    }

    /**
     * Compute the input sum and activation of each neuron in the layer and return
     * an activation matrix.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $input
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $this->input = $input;

        $this->computed = $this->weights->w()->dot($input);

        return $this->computed;
    }

    /**
     * Compute the inferential activations of each neuron in the layer.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $input
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $this->weights->w()->dot($input);
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
        if (is_null($this->input) or is_null($this->computed)) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $penalty = $this->alpha * $this->weights->w()->rowSum(0);

        $dL = [];
        $cost = 0.;

        foreach ($this->computed->row(0) as $i => $activation) {
            $expected = $labels[$i];

            $computed = $this->costFunction->compute($expected, $activation);

            $cost =+ $computed;

            $dL[$i] = $this->costFunction
                ->differentiate($expected, $activation, $computed)
                + $penalty;
        }

        $dL = new Matrix([$dL]);

        $dW = $dL->dot($this->input->transpose());

        $this->weights->update($optimizer->step($this->weights, $dW));

        unset($this->input, $this->computed);

        return [function () use ($dL) {
            return $this->weights->w()->transpose()->dot($dL);
        }, $cost];
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
