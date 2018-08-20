<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Parameter;
use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\ActivationFunctions\Rectifier;
use Rubix\ML\NeuralNet\ActivationFunctions\HyperbolicTangent;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use InvalidArgumentException;
use RuntimeException;

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
class Dense implements Hidden, Parametric
{
    const ROOT_2 = 1.41421356237;

    /**
     * The width of the layer. i.e. the number of neurons.
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
     * Should we add a bias neuron?
     *
     * @var bool
     */
    protected $bias;

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
     * The memoized output activations matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix|null
     */
    protected $computed;

    /**
     * @param  int  $neurons
     * @param  \Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction  $activationFunction
     * @param  bool  $bias
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $neurons, ActivationFunction $activationFunction, bool $bias = true)
    {
        if ($neurons < 1) {
            throw new InvalidArgumentException('The number of neurons cannot be'
                . ' less than 1.');
        }

        $this->neurons = $neurons;
        $this->activationFunction = $activationFunction;
        $this->bias = $bias;
        $this->weights = new Parameter(new Matrix([]));
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return $this->bias ? $this->neurons + 1 : $this->neurons;
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
        if ($this->activationFunction instanceof HyperbolicTangent) {
            $r = (6 / $fanIn) ** 0.25;
        } else if ($this->activationFunction instanceof Rectifier) {
            $r = (6 / $fanIn) ** (1 / self::ROOT_2);
        } else  {
            $r = sqrt(6 / $fanIn);
        }

        $min = (int) round(-$r * self::PHI);
        $max = (int) round($r * self::PHI);

        $fanOut =  $this->width();

        $w = [[]];

        for ($i = 0; $i < $fanOut; $i++) {
            for ($j = 0; $j < $fanIn; $j++) {
                $w[$i][$j] = rand($min, $max) / self::PHI;
            }
        }

        $this->weights = new Parameter(new Matrix($w));

        return $fanOut;
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

        $this->z = $temp = $this->weights->w()->multiply($input);

        if ($this->bias === true) {
            $temp = $temp->rowExclude($this->z->getM() - 1);
        }

        $activations = $this->activationFunction->compute($temp);

        if ($this->bias === true) {
            $biases = MatrixFactory::one(1, $this->z->getN());

            $activations = $activations->augmentBelow($biases);
        }

        $this->computed = $activations;

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
        $z = $this->weights->w()->multiply($input);

        if ($this->bias === true) {
            $z = $z->rowExclude($z->getM() - 1);
        }

        $activations = $this->activationFunction->compute($z);

        if ($this->bias === true) {
            $biases = MatrixFactory::one(1, $z->getN());

            $activations = $activations->augmentBelow($biases);
        }

        return $activations;
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
        if (is_null($this->input) or is_null($this->z) or is_null($this->computed)) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $errors = $this->activationFunction
            ->differentiate($this->z, $this->computed)
            ->hadamardProduct($prevErrors());

        $gradient = $errors->multiply($this->input->transpose());

        $step = $optimizer->step($this->weights, $gradient);

        $this->weights->update($step);

        unset($this->input, $this->z, $this->computed);

        return function () use ($errors) {
            return $this->weights->w()->transpose()->multiply($errors);
        };
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
