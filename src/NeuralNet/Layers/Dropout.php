<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;
use RuntimeException;

/**
 * Dropout
 *
 * Dropout layers temporarily disable neurons during each training pass. Dropout
 * is a regularization technique for reducing overfitting in neural networks
 * by preventing complex co-adaptations on training data. It is a very efficient
 * way of performing model averaging with neural networks.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Dropout implements Hidden, Nonparametric
{
    /**
     * The ratio of neurons that are dropped during each training pass.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The scaling coefficient.
     *
     * @var float
     */
    protected $scale;

    /**
     * The width of the layer.
     *
     * @var int
     */
    protected $width;

    /**
     * The memoized dropout mask.
     *
     * @var \MathPHP\LinearAlgebra\Matrix|null
     */
    protected $mask;

    /**
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(float $ratio = 0.5)
    {
        if ($ratio <= 0.0 or $ratio >= 1.0) {
            throw new InvalidArgumentException('Dropout ratio must be between 0'
                . ' and 1.0.');
        }

        $this->ratio = $ratio;
        $this->scale = 1.0 / (1.0 - $ratio);
        $this->width = 0;
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return $this->width;
    }

    /**
     * Initialize the layer.
     *
     * @param  int  $fanIn
     * @return int
     */
    public function init(int $fanIn) : int
    {
        $this->width = $fanIn;

        return $fanIn;
    }

    /**
     * Compute the input sum and activation of each nueron in the layer and
     * return an activation matrix.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $input
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $m = $input->getM();
        $n = $input->getN();

        $mask = MatrixFactory::zero($m, $n)->map(function ($value) {
            return (rand(0, self::PHI) / self::PHI) > $this->ratio
                ? $this->scale : 0.0;
        });

        $activations = $input->hadamardProduct($mask);

        $this->mask = $mask;

        return $activations;
    }

    /**
     * Compute the inferential activations of each neuron in the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $input
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $input;
    }

    /**
     * Calculate the errors and gradients of the layer and update the parameters.
     *
     * @param  callable  $prevErrors
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @throws \RuntimeException
     * @return callable
     */
    public function back(callable $prevErrors, Optimizer $optimizer) : callable
    {
        if (is_null($this->mask)) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $errors = call_user_func($prevErrors)->hadamardProduct($this->mask);

        unset($this->mask);

        return function () use ($errors) {
            return $errors;
        };
    }
}
