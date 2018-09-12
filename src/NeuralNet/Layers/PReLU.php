<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;
use RuntimeException;

/**
 * PReLU
 *
 * The PReLU layer uses ReLU activation function's whose leakage coefficients
 * are parameterized and optimized on a per neuron basis along with the weights
 * and biases.
 *
 * References:
 * [1] K. He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level
 * Performance on ImageNet Classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PReLU implements Hidden, Parametric
{
    /**
     * The width of the layer. i.e. the number of neurons.
     *
     * @var int
     */
    protected $neurons;

    /**
     * The value to initialize the alpha (leakage) parameters with.
     *
     * @var float
     */
    protected $initial;

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
     * The parameterized leakage coeficients.
     *
     * @var \Rubix\ML\NeuralNet\Parameter
     */
    protected $alphas;

    /**
     * The memoized input matrix.
     *
     * @var \Rubix\ML\Other\Structures\Matrix|null
     */
    protected $input;

    /**
     * The memoized z matrix.
     *
     * @var \Rubix\ML\Other\Structures\Matrix|null
     */
    protected $z;

    /**
     * The memoized output activations matrix.
     *
     * @var \Rubix\ML\Other\Structures\Matrix|null
     */
    protected $computed;

    /**
     * @param  int  $neurons
     * @param  float  $initial
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $neurons, float $initial = 0.25)
    {
        if ($neurons < 1) {
            throw new InvalidArgumentException('The number of neurons cannot be'
                . ' less than 1.');
        }

        if ($initial < 0. or $initial > 1.) {
            throw new InvalidArgumentException('Initial leakage parameter must'
                . ' be between 0 and 1.');
        }

        $this->neurons = $neurons;
        $this->initial = $initial;
        $this->weights = new Parameter(Matrix::empty());
        $this->biases = new Parameter(Matrix::empty());
        $this->alphas = new Parameter(Matrix::empty());
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return $this->neurons;
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
        $scale = (6 / $fanIn) ** (1. / sqrt(2));

        $w = Matrix::uniform($this->neurons, $fanIn)
            ->multiplyScalar($scale);

        $b = Matrix::zeros($this->neurons, 1);
        $a = Matrix::full($this->initial, $this->neurons, 1);

        $this->weights = new Parameter($w);
        $this->biases = new Parameter($b);
        $this->alphas = new Parameter($a);

        return $this->neurons;
    }

    /**
     * Compute the input sum and activation of each neuron in the layer and
     * return an activation matrix.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $input
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $this->input = $input;

        $this->z = $this->weights->w()->dot($input)
            ->add($this->biases->w()->repeat(1, $input->n()));

        $this->computed = $this->compute($this->z);

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
        $z = $this->weights->w()->dot($input)
            ->add($this->biases->w()->repeat(1, $input->n()));

        return $this->compute($z);
    }

    /**
     * Calculate the gradients and update the parameters of the layer.
     *
     * @param  callable  $prevGradients
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @throws \RuntimeException
     * @return callable
     */
    public function back(callable $prevGradients, Optimizer $optimizer) : callable
    {
        if (is_null($this->input) or is_null($this->z) or is_null($this->computed)) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $dOut = $prevGradients();

        $dA = $this->differentiate($this->z, $this->computed)
            ->multiply($dOut);

        $dZ = $this->z->map(function ($value) {
            return $value > 0. ? 0. : $value;
        });

        $dAlpha = $dOut->multiply($dZ)->sum()->asColumnMatrix();

        $dW = $dA->dot($this->input->transpose());
        $dB = $dA->sum()->asColumnMatrix();

        $w = $this->weights->w();

        $this->alphas->update($optimizer->step($this->alphas, $dAlpha));
        $this->weights->update($optimizer->step($this->weights, $dW));
        $this->biases->update($optimizer->step($this->biases, $dB));

        unset($this->input, $this->z, $this->computed);

        return function () use ($w, $dA) {
            return $w->transpose()->dot($dA);
        };
    }

    /**
     * Compute the leaky ReLU activation function and return a matrix.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $z
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    protected function compute(Matrix $z) : Matrix
    {
        $alphas = $this->alphas->w()->column(0);

        $computed = [[]];

        foreach ($z as $i => $row) {
            $alpha = $alphas[$i];

            foreach ($row as $j => $value) {
                $computed[$i][$j] = $value > 0.
                    ? $value
                    : $alpha * $value;
            }
        }

        return new Matrix($computed, false);
    }

    /**
     * Calculate the derivatives of the activation function.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $z
     * @param  \Rubix\ML\Other\Structures\Matrix  $computed
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    protected function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        $alphas = $this->alphas->w()->column(0);

        $gradients = [[]];

        foreach ($z as $i => $row) {
            $alpha = $alphas[$i];

            foreach ($row as $j => $value) {
                $gradients[$i][$j] = $value > 0. ? 1. : $alpha;
            }
        }

        return new Matrix($gradients, false);
    }

    /**
     * Read the parameters and return them in an associative array.
     *
     * @return array
     */
    public function read() : array
    {
        return [
            'weights' => clone $this->weights,
            'biases' => clone $this->biases,
            'alphas' => clone $this->alphas,
        ];
    }

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @param  array  $parameters
     * @return void
     */
    public function restore(array $parameters) : void
    {
        $this->weights = $parameters['weights'];
        $this->biases = $parameters['biases'];
        $this->alphas = $parameters['alphas'];
    }
}
