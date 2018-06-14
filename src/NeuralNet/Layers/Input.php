<?php

namespace Rubix\ML\NeuralNet\Layers;

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
    protected $placeholders;

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

        $this->placeholders = $inputs;
    }

    /**
     * Return the width of the input layer.
     *
     * @var int
     */
    public function width() : int
    {
        return $this->placeholders + 1;
    }

    /**
     * @return \MathPHP\LinearAlgebra\Matrix
     */
    public function computed() : Matrix
    {
        return $this->computed;
    }

    /**
     * Initialize the layer.
     *
     * @param  \Rubix\ML\NeuralNet\Layers\Layer
     * @return void
     */
    public function initialize(Layer $previous) : void
    {
        //
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
        if ($input->getM() !== $this->placeholders) {
            throw new InvalidArgumentException('The number of feature columns'
            . ' must equal the number of input placeholders. '
            . (string) $input->getM() . ' found, '
            . (string) $this->placeholders . ' needed.');
        }

        $biases = MatrixFactory::one(1, $input->getN());

        $this->computed = $input->augmentBelow($biases);

        return $this->computed;
    }

    /**
     * Do nothing since placeholder layers do not have parameters.
     *
     * @param  \Rubix\ML\NerualNet\Layers\Layer  $next
     * @return void
     */
    public function back(Layer $next) : void
    {
        //
    }
}
