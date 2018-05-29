<?php

namespace Rubix\Engine\NeuralNet\Layers;

use MathPHP\LinearAlgebra\Matrix;
use MathPHP\LinearAlgebra\MatrixFactory;
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
     * Return the width of the input layer.
     *
     * @var int
     */
    public function width() : int
    {
        return $this->inputs + 1;
    }

    /**
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function computed() : Matrix
    {
        return $this->computed;
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
            . ' must equal the number of input nodes.');
        }

        $biases = MatrixFactory::one(1, $input->getN());

        $this->computed = $input->augmentBelow($biases);

        return $this->computed;
    }

    /**
     * Do nothing since placeholder layers do not have parameters.
     *
     * @param  \Rubix\Engine\NerualNet\Layers\Layer  $next
     * @return void
     */
    public function back(Layer $next) : void
    {
        //
    }
}
