<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\ActivationFunctions\Rectifier;
use Rubix\ML\NeuralNet\ActivationFunctions\HyperbolicTangent;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use InvalidArgumentException;

/**
 * Dense
 *
 * Dense layers are fully connected Hidden layers, meaning each neuron is
 * connected to each other neuron in the previous layer. Dense layers are able
 * to employ a variety of Activation Functions that modify the output of each
 * neuron in the layer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Dense implements Hidden
{
    /**
     * The number of neurons in this layer.
     *
     * @var int
     */
    protected $neurons;

    /**
     * The function that outputs the activation or implulse of each neuron.
     *
     * @var \Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction
     */
    protected $activationFunction;

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
     * The memoized output activations matrix.
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
     * @param  int  $neurons
     * @param  \Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction  $activationFunction
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $neurons, ActivationFunction $activationFunction)
    {
        if ($neurons < 1) {
            throw new InvalidArgumentException('The number of neurons cannot be'
                . ' less than 1.');
        }

        $this->neurons = $neurons;
        $this->activationFunction = $activationFunction;
        $this->width = $neurons + 1;
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
        if ($this->activationFunction instanceof HyperbolicTangent) {
            $r = (6 / $prevWidth) ** 0.25;
        } else if ($this->activationFunction instanceof Rectifier) {
            $r = (6 / $prevWidth) ** (1 / self::ROOT_2);
        } else  {
            $r = sqrt(6 / $prevWidth);
        }

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
     * Compute the input sum and activation of each nueron in the layer and
     * return an activation matrix.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $input
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $this->input = $input;

        $this->z = $this->weights->multiply($input);

        $biases = MatrixFactory::one(1, $input->getN());

        $this->computed = $this->activationFunction
            ->compute($this->z->rowExclude($this->z->getM() - 1))
            ->augmentBelow($biases);

        return $this->computed;
    }

    /**
     * Calculate the errors and gradients of the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $prevWeights
     * @param  \MathPHP\LinearAlgebra\Matrix  $prevErrors
     * @return array
     */
    public function back(Matrix $prevWeights, Matrix $prevErrors) : array
    {
        $errors = $this->activationFunction
            ->differentiate($this->z, $this->computed)
            ->hadamardProduct($prevWeights->transpose()->multiply($prevErrors));

        $this->gradients = $errors->multiply($this->input->transpose());

        return [$this->weights, $errors];
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

        return $steps->oneNorm();
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
