<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Xavier1;
use Rubix\ML\NeuralNet\CostFunctions\CostFunction;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax;
use InvalidArgumentException;
use RuntimeException;

/**
 * Multiclass
 *
 * The Multiclass output layer gives a joint probability estimate of a multiclass
 * classification problem using the Softmax activation function.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Multiclass implements Output
{
    /**
     * The unique class labels.
     *
     * @var array
     */
    protected $classes = [
        //
    ];

    /**
     * The L2 regularization parameter.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The weight initializer.
     *
     * @var \Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $initializer;

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
     * The memoized activation matrix.
     *
     * @var \Rubix\ML\Other\Structures\Matrix|null
     */
    protected $computed;

    /**
     * @param  array  $classes
     * @param  float  $alpha
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $classes, float $alpha = 1e-4)
    {
        $classes = array_values(array_unique($classes));

        if (count($classes) < 2) {
            throw new InvalidArgumentException('The number of unique classes'
                . ' must be 2 or more.');
        }

        if ($alpha < 0.) {
            throw new InvalidArgumentException('L2 regularization parameter'
                . ' must be be non-negative.');
        }

        $this->classes = $classes;
        $this->alpha = $alpha;
        $this->initializer = new Xavier1();
        $this->activationFunction = new Softmax();
        $this->weights = new Parameter(Matrix::empty());
        $this->biases = new Parameter(Matrix::empty());
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return count($this->classes);
    }

    /**
     * Initialize the layer by fully connecting each neuron to every input and
     * generating a random weight for each parameter/synapse in the layer.
     *
     * @param  int  $fanIn
     * @return int
     */
    public function init(int $fanIn) : int
    {
        $fanOut = $this->width();

        $w = $this->initializer->initialize($fanIn, $fanOut);

        $b = Matrix::zeros($fanOut, 1);

        $this->weights = new Parameter($w);
        $this->biases = new Parameter($b);

        return $fanOut;
    }

    /**
     * Compute the input sum and activation of each neuron in the layer and return
     * an activation matrix.
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
     * Calculate the gradients for each output neuron and update.
     *
     * @param  array  $labels
     * @param  \Rubix\ML\NeuralNet\CostFunctions\CostFunction  $costFunction
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @throws \RuntimeException
     * @return array
     */
    public function back(array $labels, CostFunction $costFunction, Optimizer $optimizer) : array
    {
        if (is_null($this->input) or is_null($this->z) or is_null($this->computed)) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $expected = [[]];

        foreach ($this->classes as $i => $class) {
            foreach ($labels as $j => $label) {
                $expected[$i][$j] = $class === $label ? 1. : 0.;
            }
        }

        $expected = new Matrix($expected, false);

        $delta = $costFunction
            ->compute($expected, $this->computed);

        $penalties = $this->weights->w()->sum()->asColumnMatrix()
            ->multiplyScalar($this->alpha)
            ->repeat(1, $this->computed->n());

        $dL = $costFunction
            ->differentiate($expected, $this->computed, $delta)
            ->add($penalties);

        $dA = $this->activationFunction
            ->differentiate($this->z, $this->computed)
            ->multiply($dL);

        $dW = $dA->dot($this->input->transpose());
        $dB = $dA->sum()->asColumnMatrix();

        $w = $this->weights->w();

        $this->weights->update($optimizer->step($this->weights, $dW));
        $this->biases->update($optimizer->step($this->biases, $dB));

        $cost = $delta->sum()->sum();

        unset($this->input, $this->z, $this->computed);

        return [function () use ($w, $dA) {
            return $w->transpose()->dot($dA);
        }, $cost];
    }

    /**
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
     * Restore the parameters of the layer.
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
