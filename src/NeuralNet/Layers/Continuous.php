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
     * The biases.
     *
     * @var \Rubix\ML\NeuralNet\Parameter
     */
    protected $biases;

    /**
     * The memoized input matrix.
     *
     * @var \Rubix\ML\Other\Structures\Matrix|null
     */
    protected $input;

    /**
     * The memoized output of the layer.
     *
     * @var \Rubix\ML\Other\Structures\Matrix|null
     */
    protected $z;

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
        $this->weights = new Parameter(Matrix::empty());
        $this->biases = new Parameter(Matrix::empty());
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
        $scale = (6 / $fanIn) ** 0.25;

        $w = Matrix::uniform(1, $fanIn)
            ->multiplyScalar($scale);

        $b = Matrix::zeros(1, 1);

        $this->weights = new Parameter($w);
        $this->biases = new Parameter($b);

        return 1;
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

        $this->z = $this->weights->w()->dot($input)
            ->add($this->biases->w()->repeat(1, $input->n()));

        return $this->z;
    }

    /**
     * Compute the inferential activations of each neuron in the layer.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $input
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $this->weights->w()->dot($input)
            ->add($this->biases->w()->repeat(1, $input->n()));
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
        if (is_null($this->input) or is_null($this->z)) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $expected = new Matrix([$labels], false);

        $delta = $this->costFunction
            ->compute($expected, $this->z);

        $penalties = $this->weights->w()->sum()->asColumnMatrix()
            ->multiplyScalar($this->alpha)
            ->repeat(1, $this->z->n());

        $dL = $this->costFunction
            ->differentiate($expected, $this->z, $delta)
            ->add($penalties);

        $dW = $dL->dot($this->input->transpose());
        $dB = $dL->mean()->asColumnMatrix();

        $w = $this->weights->w();

        $this->weights->update($optimizer->step($this->weights, $dW));
        $this->biases->update($optimizer->step($this->biases, $dB));

        $cost = $delta->sum()->sum();

        unset($this->input, $this->z);

        return [function () use ($w, $dL) {
            return $w->transpose()->dot($dL);
        }, $cost];
    }

    /**
     * @return array
     */
    public function read() : array
    {
        return [
            'weights' => clone $this->weights,
            'biases' => clone $this->biases,
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
        $this->biases = $parameters['biases'];
    }
}
