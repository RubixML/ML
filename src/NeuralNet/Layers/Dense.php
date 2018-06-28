<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\ML\NeuralNet\ActivationFunctions\Rectifier;
use Rubix\ML\NeuralNet\ActivationFunctions\HyperbolicTangent;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use InvalidArgumentException;

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
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $weights;

    /**
     * The memoized input matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $input;

    /**
     * The memoized z matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $z;

    /**
     * The memoized output activations matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $computed;

    /**
     * The memoized gradient matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $gradients;

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
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function weights() : Matrix
    {
        return $this->weights;
    }

    /**
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function gradients() : Matrix
    {
        return $this->gradients;
    }

    /**
     * Initialize the layer by fully connecting each neuron to every input and
     * generating a random weight for each parameter/synapse in the layer.
     *
     * @param  int  $width
     * @return int
     */
    public function initialize(int $width) : int
    {
        $weights = array_fill(0, $this->width,
            array_fill(0, $width, 0.0));

        if ($this->activationFunction instanceof Rectifier) {
            $r = (6 / $width) ** (1 / self::ROOT_2);
        } else if ($this->activationFunction instanceof HyperbolicTangent) {
            $r = (6 / $width) ** (1 / 4);
        } else if ($this->activationFunction instanceof Sigmoid) {
            $r = sqrt(6 / $width);
        } else {
            $r = 3;
        }

        for ($i = 0; $i < $this->width; $i++) {
            for ($j = 0; $j < $width; $j++) {
                $weights[$i][$j] = random_int((int) (-$r * 1e8),
                    (int) ($r * 1e8)) / 1e8;
            }
        }

        $this->weights = new Matrix($weights);

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
     * @param  \MathPHP\LinearAlgebra\Matrix  $weights
     * @param  \MathPHP\LinearAlgebra\Matrix  $errors
     * @return array
     */
    public function back(Matrix $weights, Matrix $errors) : array
    {
        $errors = $this->activationFunction
            ->differentiate($this->z, $this->computed)
            ->hadamardProduct($weights->transpose()->multiply($errors));

        $this->gradients = $errors->multiply($this->input->transpose());

        return [$this->weights, $errors];
    }

    /**
     * Update the parameters in the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $steps
     * @return void
     */
    public function update(Matrix $steps) : void
    {
        $this->weights = $this->weights->add($steps);
    }

    /**
     * Restore the weights of the later.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $weights
     * @return void
     */
    public function restore(Matrix $weights) : void
    {
        $this->weights = $weights;
    }
}
