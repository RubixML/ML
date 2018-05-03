<?php

namespace Rubix\Engine\NeuralNet\Layers;

use Rubix\Engine\NeuralNet\Bias;
use Rubix\Engine\NeuralNet\Input as InputNode;
use InvalidArgumentException;

class Input extends Layer
{
    /**
     * @param  int  $n
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(int $n)
    {
        if ($n < 1) {
            throw new InvalidArgumentException('The number of inputs must be greater than 0.');
        }

        parent::__construct($n + 1);

        for ($i = 0; $i < $n; $i++) {
            $this[$i] = new InputNode();
        }

        $this[count($this) - 1] = new Bias();
    }

    /**
     * Prime the input layer with the values from a sample vector.
     *
     * @param  array  $sample
     * @return void
     */
    public function prime(array $sample) : void
    {
        $column = 0;

        for ($this->rewind(); $this->valid(); $this->next()) {
            $current = $this->current();

            if ($current instanceof Bias) {
                continue 1;
            }

            $current->prime($sample[$column++]);
        }
    }
}
