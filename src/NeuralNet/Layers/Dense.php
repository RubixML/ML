<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\ActivationFunctions\Rectifier;
use Rubix\ML\NeuralNet\ActivationFunctions\HyperbolicTangent;
use Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use InvalidArgumentException;
use RuntimeException;

/**
 * Dense
 *
 * Dense layers are fully connected hidden layers, meaning each neuron is
 * connected to each other neuron in the previous layer by a weighted *synapse*.
 * Dense layers employ activation functions that control the output of each
 * neuron in the layer.
 *
 * References:
 * [1] X. Glorot et al. (2010). Understanding the Difficulty of Training Deep
 * Feedforward Neural Networks.
 * [2] K. He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level
 * Performance on ImageNet Classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Dense implements Hidden, Parametric
{
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
        $this->weights = new Parameter(new Matrix([[]]));
        $this->biases = new Parameter(Matrix::ones($neurons, 1));
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
        if ($this->activationFunction instanceof Rectifier) {
            $scale = (6 / $fanIn) ** (1. / sqrt(2));
        } else if ($this->activationFunction instanceof HyperbolicTangent) {
            $scale = (6 / $fanIn) ** 0.25;
        } else  {
            $scale = sqrt(6 / $fanIn);
        }

        $w = Matrix::uniform($this->neurons, $fanIn)
            ->multiplyScalar($scale);

        $this->weights = new Parameter($w);

        return $this->neurons;
    }

    /**
     * Compute the input sum and activation of each nueron in the layer and
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

        $this->computed = $this->activationFunction->compute($this->z);

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

        return $this->activationFunction->compute($z);
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

        $dA = $this->activationFunction
            ->differentiate($this->z, $this->computed)
            ->multiply($prevGradients());

        $dW = $dA->dot($this->input->transpose());
        $dB = $dA->mean()->asColumnMatrix();

        $w = $this->weights->w();

        $this->weights->update($optimizer->step($this->weights, $dW));
        $this->biases->update($optimizer->step($this->biases, $dB));

        unset($this->input, $this->z, $this->computed);

        return function () use ($w, $dA) {
            return $w->transpose()->dot($dA);
        };
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
    }
}
