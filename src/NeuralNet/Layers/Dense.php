<?php

namespace Rubix\Engine\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use Rubix\Engine\NeuralNet\ActivationFunctions\Sigmoid;
use Rubix\Engine\NeuralNet\ActivationFunctions\Rectifier;
use Rubix\Engine\NeuralNet\ActivationFunctions\HyperbolicTangent;
use Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction;
use InvalidArgumentException;

class Dense implements Hidden, Parametric
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
     * @var \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction
     */
    protected $activationFunction;

    /**
     * The previous layer in the network.
     *
     * @var \Rubix\Engine\NeuralNet\Layers\Layer
     */
    protected $previous;

    /**
     * The weight matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $weights;

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
     * The memoized error matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $errors;

    /**
     * The memoized gradient matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
     */
    protected $gradients;

    /**
     * @param  int  $neurons
     * @param  \Rubix\Engine\NeuralNet\ActivationFunctions\ActivationFunction  $activationFunction
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
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return $this->neurons + 1;
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
    public function computed() : Matrix
    {
        return $this->computed;
    }

    /**
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function errors() : Matrix
    {
        return $this->errors;
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
     * @param  \Rubix\Engine\NeuralNet\Layers\Layer  $previous
     * @return void
     */
    public function initialize(Layer $previous) : void
    {
        $weights = array_fill(0, $this->width(),
            array_fill(0, $previous->width(), 0.0));

        if ($this->activationFunction instanceof Rectifier) {
            $r = pow(6 / $previous->width(), 1 / self::ROOT_2);
        } else if ($this->activationFunction instanceof HyperbolicTangent) {
            $r = pow(6 / $previous->width(), 1 / 4);
        } else if ($this->activationFunction instanceof Sigmoid) {
            $r = sqrt(6 / $previous->width());
        } else { $r = 3; }

        for ($i = 0; $i < $this->width(); $i++) {
            for ($j = 0; $j < $previous->width(); $j++) {
                $weights[$i][$j] = random_int(-$r * 1e8, $r * 1e8) / 1e8;
            }
        }

        $this->weights = MatrixFactory::create($weights);
        $this->previous = $previous;
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
        $activations = $this->previous->forward($input);

        $this->z = $this->weights->multiply($activations);

        $biases = MatrixFactory::one(1, $activations->getN());

        $this->computed = $this->activationFunction
            ->compute($this->z->rowExclude($this->z->getM() - 1))
            ->augmentBelow($biases);

        return $this->computed;
    }

    /**
     * Calculate the errors and gradients of the layer.
     *
     * @param  \Rubix\Engine\NerualNet\Layers\Layer  $next
     * @return void
     */
    public function back(Layer $next) : void
    {
        $this->errors = $this->activationFunction
            ->differentiate($this->z, $this->computed)
            ->hadamardProduct($next->weights()->transpose()
                ->multiply($next->errors()));

        $this->gradients = $this->errors
            ->multiply($this->previous->computed()->transpose());

        $this->previous->back($this);
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
