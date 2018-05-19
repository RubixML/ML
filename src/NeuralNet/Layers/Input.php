<?php

namespace Rubix\Engine\NeuralNet\Layers;

use InvalidArgumentException;

class Input implements Layer
{
    /**
     * The number of input nodes. i.e. feature placeholders.
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
     * Return the width of the layer.
     *
     * @var int
     */
    public function width() : int
    {
        return $this->inputs + 1;
    }

    /**
     * Just return the input vector adding a bias since the input layer does not
     * have any paramters.
     *
     * @param  array  $sample
     * @return array
     */
    public function forward(array $sample) : array
    {
        if (count($sample) !== $this->inputs) {
            throw new InvalidArgumentException('The number of sample features'
            . ' must equal the number of inputs.');
        }

        $sample[] = 1.0;

        return $sample;
    }
}
