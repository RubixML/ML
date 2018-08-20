<?php

namespace Rubix\ML\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
use InvalidArgumentException;

/**
 * Placeholder
 *
 * The Placeholder input layer serves to represent future input values as well
 * as add a bias to the forward pass.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Placeholder implements Input
{
    /**
     * The number of input nodes. i.e. feature inputs.
     *
     * @var int
     */
    protected $inputs;

    /**
     * @param  int  $inputs
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $inputs)
    {
        if ($inputs < 1) {
            throw new InvalidArgumentException('The number of input nodes must'
            . ' be greater than 0.');
        }

        $this->inputs = $inputs;
    }

    /**
     * Return the width of the input layer.
     *
     * @var int
     */
    public function width() : int
    {
        return $this->inputs + 1;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param  int  $fanIn
     * @return int
     */
    public function init(int $fanIn) : int
    {
        return $this->width();
    }

    /**
     * Just return the input vector adding a bias since the input layer does not
     * have any paramters.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $input
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        if ($input->getM() !== $this->inputs) {
            throw new InvalidArgumentException('The number of feature columns'
                . ' must equal the number of input inputs. '
                . (string) $input->getM() . ' found, '
                . (string) $this->inputs . ' needed.');
        }

        $biases = MatrixFactory::one(1, $input->getN());

        return $input->augmentBelow($biases);
    }

    /**
     * Compute the inferential activations of each neuron in the layer.
     *
     * @param  \MathPHP\LinearAlgebra\Matrix  $input
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $this->forward($input);
    }
}
