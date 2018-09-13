<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use InvalidArgumentException;
use RuntimeException;

/**
 * Dense
 *
 * Dense layers are fully connected hidden layers, meaning each neuron is
 * connected to each other neuron in the previous layer by a weighted *synapse*.
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
     * The weight initializer.
     *
     * @var \Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $initializer;

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
     * @param  int  $neurons
     * @param  \Rubix\ML\NeuralNet\Initializers\Initializer  $initializer
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $neurons, Initializer $initializer = null)
    {
        if ($neurons < 1) {
            throw new InvalidArgumentException('The number of neurons cannot be'
                . ' less than 1.');
        }

        if (is_null($initializer)) {
            $initializer = new He();
        }

        $this->neurons = $neurons;
        $this->initializer = $initializer;
        $this->weights = new Parameter(Matrix::empty());
        $this->biases = new Parameter(Matrix::empty());
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
        $fanOut = $this->width();

        $w = $this->initializer->initialize($fanIn, $fanOut);

        $b = Matrix::zeros($fanOut, 1);

        $this->weights = new Parameter($w);
        $this->biases = new Parameter($b);

        return $fanOut;
    }

    /**
     * Compute the input sum and activation of each neuron in the layer and
     * return an activation matrix.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $input
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $this->input = $input;

        return $this->weights->w()->dot($input)
            ->add($this->biases->w()->repeat(1, $input->n()));
    }

    /**
     * Compute the inferential activations of each neuron in the layer.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $input
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $this->weights->w()->dot($input)
            ->add($this->biases->w()->repeat(1, $input->n()));
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
        if (is_null($this->input)) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $dOut = $prevGradients();

        $dW = $dOut->dot($this->input->transpose());
        $dB = $dOut->sum()->asColumnMatrix();

        $w = $this->weights->w();

        $this->weights->update($optimizer->step($this->weights, $dW));
        $this->biases->update($optimizer->step($this->biases, $dB));

        unset($this->input);

        return function () use ($w, $dOut) {
            return $w->transpose()->dot($dOut);
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
