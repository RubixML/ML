<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
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
class Dropout extends Dense
{
    const PHI = 100000000;

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
     * The memoized dropout mask.
     *
     * @var \MathPHP\LinearAlgebra\Matrix|null
     */
    protected $mask;

    /**
     * @param  int  $neurons
     * @param  \Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction  $activationFunction
     * @param  float  $ratio
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $neurons, ActivationFunction $activationFunction, float $ratio = 0.5)
    {
        if ($ratio <= 0.0 or $ratio >= 1.0) {
            throw new InvalidArgumentException('Dropout ratio must be between 0'
                . ' and 1.0.');
        }

        $this->ratio = $ratio;
        $this->scale = 1.0 / (1.0 - $this->ratio);

        parent::__construct($neurons, $activationFunction);
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
        $this->input = $input;

        $this->z = $this->weights->w()->multiply($input);

        $m = $this->z->getM() - 1;
        $n = $this->z->getN();

        $mask = MatrixFactory::zero($m, $n)->map(function ($value) {
            return (rand(0, self::PHI) / self::PHI) > $this->ratio
                ? $this->scale : 0.0;
        });

        $biases = MatrixFactory::one(1, $n);

        $temp = $this->z->rowExclude($m);

        $this->computed = $this->activationFunction->compute($temp)
            ->hadamardProduct($mask)
            ->augmentBelow($biases);

        $this->mask = $mask->augmentBelow($biases);

        return $this->computed;
    }

    /**
     * Calculate the errors and gradients of the layer and update the parameters.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $prevWeights
     * @param  \MathPHP\LinearAlgebra\Matrix  $prevErrors
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @throws \RuntimeException
     * @return array
     */
    public function back(Matrix $prevWeights, Matrix $prevErrors, Optimizer $optimizer) : array
    {
        if (is_null($this->input) or is_null($this->z) or is_null($this->computed) or is_null($this->mask)) {
                throw new RuntimeException('Must perform forward pass before'
                    . ' backpropagating.');
            }

        $errors = $this->activationFunction
            ->differentiate($this->z, $this->computed)
            ->hadamardProduct($prevWeights->transpose()->multiply($prevErrors))
            ->hadamardProduct($this->mask);

        $gradient = $errors->multiply($this->input->transpose());

        $step = $optimizer->step($this->weights, $gradient);

        $this->weights->update($step);

        unset($this->input, $this->z, $this->computed, $this->mask);

        return [$this->weights->w(), $errors, $step->maxNorm()];
    }
}
