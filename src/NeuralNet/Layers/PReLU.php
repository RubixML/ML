<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\Tensor\Matrix;
use Rubix\ML\NeuralNet\Parameter;
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
     * The value to initialize the alpha (leakage) parameters with.
     *
     * @var float
     */
    protected $initial;

    /**
     * The width of the layer.
     *
     * @var int|null
     */
    protected $width;

    /**
     * The parameterized leakage coeficients.
     *
     * @var \Rubix\ML\NeuralNet\Parameter|null
     */
    protected $alphas;

    /**
     * The memoized input matrix.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $input;

    /**
     * The memoized activation matrix.
     *
     * @var \Rubix\Tensor\Matrix|null
     */
    protected $computed;

    /**
     * @param  float  $initial
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $initial = 0.25)
    {
        if ($initial < 0. or $initial > 1.) {
            throw new InvalidArgumentException('Initial leakage parameter must'
                . ' be between 0 and 1.');
        }

        $this->initial = $initial;
    }

    /**
     * @return int|null
     */
    public function width() : ?int
    {
        return $this->width;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param  int  $fanIn
     * @return int
     */
    public function init(int $fanIn) : int
    {
        $fanOut = $fanIn;

        $a = Matrix::fill($this->initial, $fanOut, 1);

        $this->alphas = new Parameter($a);

        $this->width = $fanOut;

        return $fanOut;
    }

    /**
     * Compute the input sum and activation of each neuron in the layer and
     * return an activation matrix.
     *
     * @param  \Rubix\Tensor\Matrix  $input
     * @return \Rubix\Tensor\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $this->input = $input;

        $this->computed = $this->rectify($input);

        return $this->computed;
    }

    /**
     * Compute the inferential activations of each neuron in the layer.
     *
     * @param  \Rubix\Tensor\Matrix  $input
     * @return \Rubix\Tensor\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {   
        return $this->rectify($input);
    }

    /**
     * Calculate the gradients and update the parameters of the layer.
     *
     * @param  callable  $prevGradient
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @throws \RuntimeException
     * @return callable
     */
    public function back(callable $prevGradient, Optimizer $optimizer) : callable
    {
        if (is_null($this->alphas)) {
            throw new RuntimeException('Layer has not been initlaized.');
        }

        if (is_null($this->input) or is_null($this->computed)) {
            throw new RuntimeException('Must perform a forward pass before'
                . ' backpropagating.');
        }

        $dOut = $prevGradient();

        $dIn = $this->input->clip(-INF, 0.);

        $dAlpha = $dOut->multiply($dIn)->sum()->asColumnMatrix();

        $this->alphas->update($optimizer->step($this->alphas, $dAlpha));

        $z = $this->input;
        $computed = $this->computed;

        unset($this->input, $this->computed);

        return function () use ($z, $computed, $dOut) {
            return $this->differentiate($z, $computed)
                ->multiply($dOut);
        };
    }

    /**
     * Compute the leaky ReLU activation function and return a matrix.
     *
     * @param  \Rubix\Tensor\Matrix  $z
     * @throws \RuntimeException
     * @return \Rubix\Tensor\Matrix
     */
    protected function rectify(Matrix $z) : Matrix
    {
        if (is_null($this->alphas)) {
            throw new RuntimeException('Layer has not been initlaized.');
        }

        $alphas = $this->alphas->w()->column(0);

        $computed = [];

        foreach ($z as $i => $row) {
            $alpha = $alphas[$i];

            $activations = [];

            foreach ($row as $value) {
                $activations[] = $value > 0.
                    ? $value
                    : $alpha * $value;
            }

            $computed[] = $activations;
        }

        return Matrix::quick($computed);
    }

    /**
     * Calculate the derivatives of the activation function.
     *
     * @param  \Rubix\Tensor\Matrix  $z
     * @param  \Rubix\Tensor\Matrix  $computed
     * @throws \RuntimeException
     * @return \Rubix\Tensor\Matrix
     */
    protected function differentiate(Matrix $z, Matrix $computed) : Matrix
    {
        if (is_null($this->alphas)) {
            throw new RuntimeException('Layer has not been initlaized.');
        }

        $alphas = $this->alphas->w()->column(0);

        $gradients = [];

        foreach ($z as $i => $row) {
            $alpha = $alphas[$i];

            $gradient = [];

            foreach ($row as $value) {
                $gradient[] = $value > 0. ? 1. : $alpha;
            }

            $gradients[] = $gradient;
        }

        return Matrix::quick($gradients);
    }

    /**
     * Read the parameters and return them in an associative array.
     *
     * @throws \RuntimeException
     * @return array
     */
    public function read() : array
    {
        if (is_null($this->alphas)) {
            throw new RuntimeException('Layer has not been initlaized.');
        }
        
        return [
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
        $this->alphas = $parameters['alphas'];
    }
}
