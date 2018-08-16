<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\ActivationFunctions\SELU;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use InvalidArgumentException;
use RuntimeException;

/**
 * Alpha Dropout
 *
 * Alpha Dropout is a type of dropout layer that maintains the mean and variance
 * of the original inputs in order to ensure the self-normalizing property of
 * SELU networks with dropout. Alpha Dropout fits with SELU networks by randomly
 * setting activations to the negative saturation value of the activation
 * function at a given ratio each pass.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class AlphaDropout extends Dense
{
    const ALPHA = 1.6732632423543772848170429916717;
    const SCALE = 1.0507009873554804934193349852946;

    const ALPHA_P = -self::ALPHA * self::SCALE;

    const PHI = 1000000;

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
    protected $a;

    /**
     * The centering coefficient.
     *
     * @var float
     */
    protected $b;

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
    public function __construct(int $neurons, ActivationFunction $activationFunction = null,
                                float $ratio = 0.1)
    {
        if ($ratio <= 0.0 or $ratio >= 1.0) {
            throw new InvalidArgumentException('Dropout ratio must be between 0'
                . ' and 1.0.');
        }

        if (is_null($activationFunction)) {
            $activationFunction = new SELU(self::ALPHA, self::SCALE);
        }

        $this->ratio = $ratio;
        $this->a = ((1 - $ratio) * (1 + $ratio * self::ALPHA_P ** 2)) ** -0.5;
        $this->b = -$this->a * self::ALPHA_P * $ratio;

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
            return (rand(0, self::PHI) / self::PHI) > $this->ratio ? 1.0 : 0.0;
        });

        $saturation = $mask->map(function ($value) {
            return $value === 0.0 ? self::ALPHA_P : 0.0;
        });

        $biases = MatrixFactory::one(1, $n);

        $temp = $this->z->rowExclude($m);

        $this->computed = $this->activationFunction->compute($temp)
            ->hadamardProduct($mask)
            ->add($saturation)
            ->map(function ($activation) {
                return $this->a * $activation + $this->b;
            })
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

        return [$this->weights->w(), $errors];
    }
}
