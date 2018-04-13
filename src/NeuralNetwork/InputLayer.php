<?php

namespace Rubix\Engine\NeuralNetwork;

class InputLayer extends Layer
{
    /**
     * @param  int  $inputs
     * @return void
     */
    public function __construct(int $inputs)
    {
        parent::__construct($inputs + 1);

        for ($i = 0; $i < $inputs; $i++) {
            $this[$i] = new Input();
        }

        $this[count($this) - 1] = new Bias();
    }
}
