<?php

namespace Rubix\Engine\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use InvalidArgumentException;

class Softmax implements Output, Parametric
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

        $this->classes = $classes;
        $this->alpha = $alpha;
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return count($this->classes);
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

        $r = sqrt(6 / $previous->width());

        for ($i = 0; $i < $this->width(); $i++) {
            for ($j = 0; $j < $previous->width(); $j++) {
                $weights[$i][$j] = random_int(-$r * 1e8, $r * 1e8) / 1e8;
            }
        }

        $this->weights = MatrixFactory::create($weights);
        $this->previous = $previous;
    }

    /**
     * Compute the input sum and activation of each neuron in the layer and return
     * an activation matrix.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $input
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        $activations = $this->previous->forward($input);

        $this->z = $this->weights->multiply($activations);

        $outputs = $cache = [[]];

        foreach ($this->z->asVectors() as $i => $z) {
            for ($j = 0; $j < $z->getN(); $j++) {
                $cache[$i][$j] = exp($z[$j]);
            }

            $sigma = array_sum($cache[$i]);

            for ($j = 0; $j < $z->getN(); $j++) {
                $outputs[$j][$i] = $cache[$i][$j] / ($sigma + self::EPSILON);
            }
        }

        $this->computed = MatrixFactory::create($outputs);

        return $this->computed;
    }

    /**
     * Calculate the errors and gradients for each output neuron.
     *
     * @param  array  $labels
     * @return void
     */
    public function back(array $labels) : void
    {
        $errors = [[]];

        foreach ($labels as $i => $label) {
            foreach ($this->classes as $j => $class) {
                $expected = $class === $label ? 1.0 : 0.0;

                $errors[$j][$i] = ($expected - $this->computed[$j][$i])
                    + 0.5 * $this->alpha * array_sum($this->weights[$j]) ** 2;
            }
        }

        $this->errors = MatrixFactory::create($errors);

        $this->gradients = $this->errors
            ->multiply($this->previous->computed()->transpose());

        $this->previous->back($this);
    }

    /**
     * Return an array with the output activations for each class.
     *
     * @return array
     */
    public function activations() : array
    {
        $activations = [];

        foreach ($this->computed->getMatrix() as $i => $neuron) {
            foreach ($neuron as $j => $activation) {
                $activations[$j][$this->classes[$i]] = $activation;
            }
        }

        return $activations;
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
     * Restore the parameters in the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $weights
     * @return void
     */
    public function restore(Matrix $weights) : void
    {
        $this->weights = $weights;
    }
}
