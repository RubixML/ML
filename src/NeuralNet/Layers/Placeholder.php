<?php

namespace Rubix\ML\NeuralNet\Layers;

use Rubix\ML\Other\Structures\Matrix;
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
     * Should we add a bias node?
     *
     * @var bool
     */
    protected $bias;

    /**
     * @param  int  $inputs
     * @param  bool  $bias
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $inputs, bool $bias = true)
    {
        if ($inputs < 1) {
            throw new InvalidArgumentException('The number of input nodes must'
            . ' be greater than 0.');
        }

        $this->inputs = $inputs;
        $this->bias = $bias;
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return $this->bias ? $this->inputs + 1 : $this->inputs;
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
     * @param  \Rubix\ML\Other\Structures\Matrix  $input
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function forward(Matrix $input) : Matrix
    {
        if ($input->m() !== $this->inputs) {
            throw new InvalidArgumentException('The number of feature columns'
                . ' must equal the number of input inputs. '
                . (string) $input->m() . ' found, '
                . (string) $this->inputs . ' needed.');
        }

        $activations = $input;

        if ($this->bias === true) {
            $biases = Matrix::ones(1, $input->n());

            $activations = $activations->augmentBelow($biases);
        }

        return $activations;
    }

    /**
     * Compute the inferential activations of each neuron in the layer.
     *
     * @param  \Rubix\ML\Other\Structures\Matrix  $input
     * @return \Rubix\ML\Other\Structures\Matrix
     */
    public function infer(Matrix $input) : Matrix
    {
        return $this->forward($input);
    }
}
