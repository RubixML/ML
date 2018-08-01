<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Parameter;
use MathPHP\LinearAlgebra\Matrix;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;

/**
 * Softmax
 *
 * A generalization of the Logistic Layer, the Softmax Output Layer gives a
 * joint probability estimate of a multiclass classification problem.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Softmax implements Output
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
     * The width of the layer. i.e. the number of neurons.
     *
     * @var int
     */
    protected $width;

    /**
     * The weights.
     *
     * @var \Rubix\ML\NeuralNet\Parameter
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
     * The memoized activation matrix.
     *
     * @var \MathPHP\LinearAlgebra\Matrix
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

        if ($alpha < 0) {
            throw new InvalidArgumentException('L2 regularization parameter'
                . ' must be 0 or greater.');
        }

        $this->classes = $classes;
        $this->alpha = $alpha;
        $this->width = count($classes);
        $this->weights = new Parameter(new Matrix([]));
        $this->input = new Matrix([]);
        $this->z = new Matrix([]);
        $this->computed = new Matrix([]);
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
     * @param  int  $fanIn
     * @return int
     */
    public function init(int $fanIn) : int
    {
        $r = sqrt(6 / $fanIn);

        $min = (int) (-$r * self::PHI);
        $max = (int) ($r * self::PHI);

        $w = [[]];

        for ($i = 0; $i < $this->width; $i++) {
            for ($j = 0; $j < $fanIn; $j++) {
                $w[$i][$j] = rand($min, $max) / self::PHI;
            }
        }

        $this->weights = new Parameter(new Matrix($w));

        return $this->width;
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
        $this->input = $input;

        $this->z = $this->weights->w()->multiply($input);

        $activations = [[]];

        foreach ($this->z->asVectors() as $i => $z) {
            $cache = [];

            foreach ($z->getVector() as $j => $value) {
                $cache[$j] = exp($value);
            }

            $sigma = array_sum($cache) + self::EPSILON;

            foreach ($cache as $j => $value) {
                $activations[$j][$i] = $value / $sigma;
            }
        }

        $this->computed = new Matrix($activations);

        return $this->computed;
    }

    /**
     * Calculate the errors and gradients for each output neuron and update.
     *
     * @param  array  $labels
     * @param  \Rubix\ML\NeuralNet\Optimizers\Optimizer  $optimizer
     * @return array
     */
    public function back(array $labels, Optimizer $optimizer) : array
    {
        $w = $this->weights->w();

        $errors = [[]];

        foreach ($this->classes as $i => $class) {
            $penalty = 0.5 * $this->alpha * array_sum($w->getRow($i)) ** 2;

            foreach ($this->computed->getRow($i) as $j => $activation) {
                $expected = $class === $labels[$j] ? 1.0 : 0.0;

                $errors[$i][$j] = ($expected - $activation)
                    * ($activation * (1 - $activation))
                    + $penalty;
            }
        }

        $errors = new Matrix($errors);

        $gradients = $errors->multiply($this->input->transpose());

        $step = $optimizer->step($this->weights, $gradients);

        $this->weights->update($step);

        return [$w, $errors, $step->maxNorm()];
    }

    /**
     * Return the computed activation matrix.
     *
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function activations() : Matrix
    {
        return $this->computed->transpose();
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
